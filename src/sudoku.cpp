//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include <opencv2/opencv.hpp>

#include "cpp_utils/data.hpp"
#include "cpp_utils/stop_watch.hpp"

#define DLL_SVM_SUPPORT
#include "dll/dbn.hpp"
#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/conv_dbn.hpp"
#include "dll/test.hpp"
#include "dll/labels.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include "detector.hpp"
#include "solver.hpp"
#include "dataset.hpp"
#include "config.hpp"
#include "image_utils.hpp"
#include "utils.hpp"

namespace {

//These constants need to be sync when changing MNIST dataset and the fill colors
constexpr const std::size_t mnist_size_1 = 60000;
constexpr const std::size_t mnist_size_2 = 10000;
constexpr const std::size_t n_colors = 6;

using mixed_dbn_std_t = dll::conv_dbn_desc<
    dll::dbn_layers<
        dll::conv_rbm_desc<CELL_SIZE, 1, 21, 40,
            dll::momentum,
            dll::parallel,
            dll::visible<dll::unit_type::GAUSSIAN>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>
        >::rbm_t,
        dll::conv_rbm_desc<21, 40, 16, 40,
            dll::momentum,
            dll::parallel,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>>::rbm_t/*,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::rbm_t*/
    >, dll::svm_concatenate>::dbn_t;

using mixed_dbn_pmp_t = dll::conv_dbn_desc<
    dll::dbn_layers<
        dll::conv_rbm_mp_desc<CELL_SIZE, 1, 22, 40, 2,
            dll::momentum,
            dll::parallel,
            dll::weight_decay<dll::decay_type::L2>,
            //dll::visible<dll::unit_type::GAUSSIAN>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>
        >::rbm_t,
        dll::conv_rbm_mp_desc<11, 40, 6, 40, 2,
            dll::momentum,
            dll::parallel,
            dll::weight_decay<dll::decay_type::L2>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>>::rbm_t/*,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::rbm_t*/
    >, dll::svm_concatenate/*, dll::svm_scale*/>::dbn_t;

using mixed_dbn_pmp_big_t = dll::conv_dbn_desc<
    dll::dbn_layers<
        dll::conv_rbm_mp_desc<BIG_CELL_SIZE, 1, 32, 40, 2,
            dll::momentum,
            dll::parallel,
            dll::weight_decay<dll::decay_type::L2>,
            //dll::visible<dll::unit_type::GAUSSIAN>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>
        >::rbm_t,
        dll::conv_rbm_mp_desc<16, 40, 10, 40, 2,
            dll::momentum,
            dll::parallel,
            dll::weight_decay<dll::decay_type::L2>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>>::rbm_t/*,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::rbm_t*/
    >, dll::svm_concatenate/*, dll::svm_scale*/>::dbn_t;

using mixed_dbn_t = mixed_dbn_pmp_t;

using dbn_t = dll::dbn_desc<
    dll::dbn_layers<
        dll::rbm_desc<CELL_SIZE * CELL_SIZE, 300, dll::momentum, dll::batch_size<10>, dll::init_weights>::rbm_t,
        dll::rbm_desc<300, 300, dll::momentum, dll::batch_size<10>>::rbm_t,
        dll::rbm_desc<300, 500, dll::momentum, dll::batch_size<10>>::rbm_t,
        dll::rbm_desc<500, 9, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
    >>::dbn_t;

using dbn_p = std::unique_ptr<dbn_t>;

int command_detect(const config& conf){
    if(conf.files.empty()){
        std::cout << "Usage: sudoku detect <image>..." << std::endl;
        return -1;
    }

    bool view = conf.files.size() == 1 && conf.command != "detect_save";

    for(auto image_source_path : conf.files){
        std::cout << image_source_path << std::endl;

        auto source_image = open_image(image_source_path);

        if (!source_image.data){
            std::cout << "Invalid source_image" << std::endl;
            continue;
        }

        cv::Mat dest_image;
        detect(source_image, dest_image, conf.mixed);

        if(view){
            cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
            cv::imshow("Sudoku Grid", dest_image);

            cv::waitKey(0);
        } else {
            image_source_path.insert(image_source_path.rfind('.'), ".lines");
            imwrite(image_source_path.c_str(), dest_image);
        }
    }

    return 0;
}

template<typename Dataset>
cv::Mat fill_image(const std::string& source, Dataset& mnist_dataset, const std::vector<cv::Vec3b>& colors, bool write){
    static std::random_device rd{};
    static std::default_random_engine rand_engine{rd()};

    static std::uniform_int_distribution<std::size_t> digit_distribution(0, mnist_size_1 + mnist_size_2);
    static std::uniform_int_distribution<int> offset_distribution(-3, 3);
    static std::uniform_int_distribution<std::size_t> color_distribution(0, n_colors - 1);

    static auto digit_generator = std::bind(digit_distribution, rand_engine);
    static auto offset_generator = std::bind(offset_distribution, rand_engine);
    static auto color_generator = std::bind(color_distribution, rand_engine);

    std::cout << "Process image " << source << std::endl;

    auto source_image = open_image(source);
    auto original_image = open_image(source, false);

    if (!source_image.data || !original_image.data){
        std::cout << "Invalid source_image" << std::endl;
        return original_image;
    }

    cv::Mat dest_image = original_image.clone();

    //Detect if image was resized

    bool resized = source_image.size() != original_image.size();
    auto w_ratio = static_cast<double>(source_image.size().width) / original_image.size().width;
    auto h_ratio = static_cast<double>(source_image.size().height) / original_image.size().height;

    //Detect the grid/cells

    cv::Mat detect_dest_image;
    auto grid = detect(source_image, detect_dest_image);

    if(!grid.valid()){
        std::cout << "Invalid grid" << std::endl;
    }

    //We use the ground truth to complete/fix the detection pass

    auto data = read_data(source);

    if(!data.valid){
        std::cout << "The ground truth data is not valid" << std::endl;
    }

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            auto& cell = grid(j, i);

            cell.value() = data.results[i][j];
            cell.m_empty = !cell.value();
        }
    }

    if(!is_valid(grid)){
        std::cout << "The grid is not valid" << std::endl;
    }

    //Solve the grid (if it fails (bad detection/ground truth), random fill)

    if(!solve(grid)){
        std::cout << "The grid is not solvable" << std::endl;
        solve_random(grid);
    }

    //Update the ground truth

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            data.results[i][j] = grid(j, i).value();
        }
    }

    //Pick a random color for the whole sudoku
    const auto& fill_color = colors[color_generator()];

    for(auto& cell : grid.cells){
        if(cell.empty()){
            auto& bounding_rect = cell.bounding;

            //Note to self: This is pretty stupid (theoretical possible infinite loop)
            auto r = digit_generator();
            while(true){
                auto label = r < mnist_size_1 ? mnist_dataset.training_labels[r] : mnist_dataset.test_labels[r - mnist_size_1];

                if(label == cell.value()){
                    break;
                }

                r = digit_generator();
            }

            //Get the digit from MNIST

            auto image = r < mnist_size_1 ? mnist_dataset.training_images[r] : mnist_dataset.test_images[r - mnist_size_1];

            cv::Mat image_mat(28, 28, CV_8U);
            for(std::size_t xx = 0; xx < 28; ++xx){
                for(std::size_t yy = 0; yy < 28; ++yy){
                    image_mat.at<uchar>(cv::Point(xx, yy)) = image[yy * 28 + xx];
                }
            }

            //Center the digit inside the cell (plus some random offsets)

            auto x_start = offset_generator() + bounding_rect.x + (bounding_rect.width - 28) / 2;
            auto y_start = offset_generator() + bounding_rect.y + (bounding_rect.height - 28) / 2;

            //Apply reverse ratio
            if(resized){
                cv::Mat resized;
                cv::resize(image_mat, resized, cv::Size(), 1.0 / w_ratio, 1.0 / h_ratio, CV_INTER_CUBIC);
                image_mat = resized;

                x_start *= (1.0 / w_ratio);
                y_start *= (1.0 / h_ratio);
            }

            //Draw the digit

            for(int xx = 0; xx < image_mat.size().width; ++xx){
                for(int yy = 0; yy < image_mat.size().height; ++yy){
                    auto mnist_color = image_mat.at<uchar>(cv::Point(xx, yy));

                    if(mnist_color > 40){
                        auto& color = dest_image.at<cv::Vec3b>(cv::Point(xx + x_start, yy + y_start));

                        color[0] = fill_color[0];
                        color[1] = fill_color[1];
                        color[2] = fill_color[2];
                    }
                }
            }

            //Apply a light blur on the drawed digit

            cv::Rect mnist_rect(x_start, y_start, image_mat.size().width, image_mat.size().height);
            cv::GaussianBlur(dest_image(mnist_rect), dest_image(mnist_rect), cv::Size(0,0), 1);
        }
    }

    if(write){
        std::string dest(source);
        dest.insert(dest.rfind('.'), ".mixed");
        imwrite(dest.c_str(), dest_image);

        write_data(dest, data);
    }

    return dest_image;
}

int command_fill(const config& conf){
    if(conf.files.empty()){
        std::cout << "Usage: sudoku fill <image>..." << std::endl;
        return -1;
    }

    //Create colors for the generation

    std::vector<cv::Vec3b> colors;
    colors.emplace_back(25, 25, 25);
    colors.emplace_back(30, 30, 30);
    colors.emplace_back(25, 25, 145);
    colors.emplace_back(35, 45, 125);
    colors.emplace_back(145, 25, 25);
    colors.emplace_back(120, 40, 40);

    std::cout << "Load MNIST Dataset" << std::endl;
    auto mnist_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    if(mnist_dataset.training_images.empty() || mnist_dataset.test_images.empty()){
        std::cout << "Impossible to load MNIST images" << std::endl;
        return -1;
    }

    if(mnist_dataset.training_images.size() != mnist_size_1 || mnist_dataset.test_images.size() != mnist_size_2){
        std::cout << "Constants mnist_size_1 and mnist_size_2 need to be updated!" << std::endl;
        return -1;
    }

    if(colors.size() != n_colors){
        std::cout << "Constant n_colors needs to be updated!" << std::endl;
        return -1;
    }

    bool view = conf.files.size() == 1 && conf.command != "fill_save";

    for(auto& image_source_path : conf.files){
        auto dest_image = fill_image(image_source_path, mnist_dataset, colors, !view);

        if(view){
            cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
            cv::imshow("Sudoku Grid", dest_image);

            cv::waitKey(0);
        }
    }

    return 0;
}

int command_train(const config& conf){
    std::cout << mixed_dbn_t::full_output_size() << std::endl;
    auto ds = get_dataset(conf);

    std::cout << "Train with " << ds.source_grids.size() << " sudokus" << std::endl;

    std::cout << "Train with " << ds.training_images.size() << " cells" << std::endl;

    if(conf.test){
        std::cout << "Test with " << ds.test_images.size() << " cells" << std::endl;
    }

    if(conf.mixed){
        auto dbn = std::make_unique<mixed_dbn_t>();
        dbn->display();

        dbn->layer<1>().learning_rate *= 2.0;

        dbn->layer<0>().pbias = 0.10;

        dbn->layer<1>().pbias_lambda = 2;
        dbn->layer<1>().pbias = 0.08;

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(ds.training_images, 25); //TODO Increase

        svm_parameter parameters = dll::default_svm_parameters();

        parameters.svm_type = C_SVC;
        parameters.kernel_type = RBF;
        parameters.probability = 1;
        parameters.C = 2.8;
        parameters.gamma = 0.01;

        if(conf.grid){
            //Normal grid search
            //dbn->svm_grid_search(ds.training_images, ds.training_labels);

            //Coarser grid search

            svm::rbf_grid coarse_grid;
            coarse_grid.c_first = 0.1;
            coarse_grid.c_last = 20;
            coarse_grid.c_steps = 10;
            coarse_grid.c_search = svm::grid_search_type::LINEAR;

            coarse_grid.gamma_first = 1e-4;
            coarse_grid.gamma_last = 2e-2;
            coarse_grid.gamma_steps = 7;
            coarse_grid.gamma_search = svm::grid_search_type::EXP;

            dbn->svm_grid_search(ds.training_images, ds.training_labels, 4, coarse_grid);
        } else {
            dbn->svm_train(ds.training_images, ds.training_labels, parameters);

            auto training_error = dll::test_set(dbn, ds.training_images, ds.training_labels, dll::svm_predictor());
            std::cout << "training_error:" << training_error  << std::endl;

            if(conf.test){
                auto test_error = dll::test_set(dbn, ds.test_images, ds.test_labels, dll::svm_predictor());
                std::cout << "test_error:" << test_error << std::endl;
            }

            std::ofstream os("cdbn.dat", std::ofstream::binary);
            dbn->store(os);
        }
    } else {
        auto dbn = std::make_unique<dbn_t>();
        dbn->display();

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(ds.training_images, 20);

        std::cout << "Start fine-tuning" << std::endl;
        dbn->fine_tune(ds.training_images, ds.training_labels, 10, 100);

        std::cout << "training_error:" << dll::test_set(dbn, ds.training_images, ds.training_labels, dll::predictor()) << std::endl;

        if(conf.test){
            std::cout << "test_error:" << dll::test_set(dbn, ds.test_images, ds.test_labels, dll::predictor()) << std::endl;
        }

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);
    }

    return 0;
}

int command_recog(const config& conf){
    std::string image_source_path(conf.files.front());

    std::string dbn_path = conf.mixed ? "cdbn.dat" : "final.dat";
    if(conf.files.size() > 1){
        dbn_path = conf.files[1];
    }

    std::ifstream is(dbn_path, std::ofstream::binary);
    if(!is.is_open()){
        std::cerr << dbn_path << " does not exist or is not readable" << std::endl;
        return 1;
    }

    cv::Mat source_image;
    cv::Mat dest_image;

    sudoku_grid grid;

    if(conf.command == "recog"){
        source_image = open_image(image_source_path);

        if (!source_image.data){
            std::cerr << "Invalid source_image" << std::endl;
            return 1;
        }

        grid = detect(source_image, dest_image, conf.mixed);
    } else if(conf.command == "recog_binary"){
        std::ifstream is(image_source_path);

        std::size_t rows = 0;
        std::size_t columns = 0;

        std::string line;
        while (std::getline(is, line)){
            std::size_t local_columns = 0;
            for(std::size_t i = 0; i < line.size(); ++i){
                if(line[i] == '1' || line[i] == '0'){
                    if(i+1 < line.size() && line[i+1] != ','){
                        std::cout << "Invalid format of the binary file" << std::endl;
                        return 1;
                    }
                    ++local_columns;
                    ++i;
                } else {
                    std::cout << "Invalid format of the binary file" << std::endl;
                    return 1;
                }
            }

            if(columns == 0){
                columns = local_columns;
            } else if(columns != local_columns){
                std::cout << "Invalid format of the binary file" << std::endl;
                return 1;
            }

            ++rows;
        }

        is.clear();
        is.seekg(0, std::ios::beg);

        source_image.create(rows, columns, CV_8U);

        int i = 0;

        while (std::getline(is, line)){
            std::size_t j = 0;
            for(std::size_t x = 0; x < line.size(); ++x){
                if(line[x] == '1' || line[x] == '0'){
                    source_image.at<uint8_t>(i,j) = line[x] == '1' ? 255 : 0;
                    ++j;
                }
            }

            ++i;
        }

        grid = detect_binary(source_image, dest_image, conf.mixed);
    }

    if(!grid.valid()){
        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                std::cout << "0 ";
            }
            std::cout << std::endl;
        }
    } else {
        std::vector<std::tuple<std::size_t, std::size_t, double>> next;

        std::array<std::array<int, 9>, 9> matrix;

        if(conf.mixed){
            auto dbn = std::make_unique<mixed_dbn_t>();
            dbn->load(is);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    matrix[i][j] = dbn->svm_predict(grid(j, i).image(conf));
                }
            }
        } else {
            auto dbn = std::make_unique<dbn_t>();
            dbn->load(is);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    auto& cell = grid(j, i);

                    std::size_t answer;
                    if(cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image(conf));
                        answer = dbn->predict_label(weights)+1;
                        for(std::size_t x = 0; x < weights.size(); ++x){
                            if(answer != x + 1 && weights(x) > 1e-5){
                                next.push_back(std::make_tuple(i * 9 + j, x + 1, weights(x)));
                            }
                        }
                    }
                    matrix[i][j] = answer;
                }
            }
        }

        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }

        if(!next.empty()){
            std::sort(next.begin(), next.end(), [](auto& lhs, auto& rhs){
                return std::get<2>(lhs) > std::get<2>(rhs);
                });

            for(std::size_t n = 0; n < next.size() && n < 5; ++n){
                std::cout << std::endl;

                auto change = next[n];

                for(size_t i = 0; i < 9; ++i){
                    for(size_t j = 0; j < 9; ++j){
                        if(std::get<0>(change) == i * 9 + j){
                            std::cout << std::get<1>(change) << " ";
                        } else {
                            std::cout << matrix[i][j] << " ";
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    return 0;
}

int command_test(const config& conf){
    auto ds = get_dataset(conf);

    std::cout << "Test with " << ds.source_grids.size() << " sudokus" << std::endl;
    std::cout << "Test with " << ds.all_images.size() << " cells" << std::endl;

    std::size_t sudoku_hits = 0;
    std::size_t cell_hits = 0;
    std::size_t zero_errors = 0;
    std::size_t dbn_errors = 0;

    if(conf.mixed){
        auto dbn = std::make_unique<mixed_dbn_t>();

        dbn->display();

        std::ifstream is("cdbn.dat", std::ofstream::binary);
        dbn->load(is);

        for(std::size_t i = 0; i < ds.source_grids.size(); ++i){
            const auto& grid = ds.source_grids[i];

            if(!conf.quiet){
                std::cout << grid.source_image_path << std::endl;
            }

            std::size_t local_hits = 0;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    auto& cell = grid(j,i);
                    auto image = cell.image(conf);

                    preprocess(image, conf);

                    auto answer = dbn->svm_predict(image) + 1;
                    auto correct = cell.correct();

                    if(answer == correct){
                        ++local_hits;
                    } else {
                        ++dbn_errors;

                        if(!conf.quiet){
                            std::cout << "ERROR: " << std::endl;
                            std::cout << "\t where: " << i << ":" << j << std::endl;
                            std::cout << "\t answer: " << static_cast<size_t>(answer) << std::endl;
                            std::cout << "\t was: " << static_cast<size_t>(correct) << std::endl;
                        }
                    }
                }
            }

            if(local_hits == 81){
                ++sudoku_hits;
            }

            cell_hits += local_hits;
        }
    } else {
        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::ifstream is("dbn.dat", std::ofstream::binary);
        dbn->load(is);

        auto error_rate = dll::test_set(dbn, ds.all_images, ds.all_labels, dll::predictor());

        std::cout << std::endl;
        std::cout << "DBN Error rate (normal): " << 100.0 * error_rate << "%" << std::endl;

        for(std::size_t i = 0; i < ds.source_grids.size(); ++i){
            const auto& grid = ds.source_grids[i];

            std::cout << grid.source_image_path << std::endl;

            std::size_t local_hits = 0;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;
                    auto correct = grid(j,i).correct();
                    auto& cell_mat = grid(j,i).mat(conf);

                    auto fill = fill_factor(cell_mat);

                    auto weights = dbn->activation_probabilities(grid(j,i).image(conf));
                    if(fill == 1.0f){
                        answer = 0;
                    } else {
                        answer = dbn->predict_label(weights)+1;
                    }

                    if(answer == correct){
                        ++local_hits;
                    } else {
                        if(!answer || !correct){
                            ++zero_errors;
                        } else {
                            ++dbn_errors;
                        }

                        if(!conf.quiet){
                            std::cout << "ERROR: " << std::endl;
                            std::cout << "\t where: " << i << ":" << j << std::endl;
                            std::cout << "\t answer: " << static_cast<size_t>(answer) << std::endl;
                            std::cout << "\t was: " << static_cast<size_t>(correct) << std::endl;
                            std::cout << "\t fill_factor: " << fill << std::endl;

                            std::cout << "\t weights: {";
                            for(std::size_t i = 0; i < weights.size(); ++i){
                                if(i > 0){
                                    std::cout << ",";
                                }
                                std::cout << weights[i];
                            }
                            std::cout << "}" << std::endl;
                        }
                    }
                }
            }

            if(local_hits == 81){
                ++sudoku_hits;
            }

            cell_hits += local_hits;
        }
    }

    auto total_s = static_cast<double>(ds.source_grids.size());
    auto total_c = total_s * 81.0;

    std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "% (" << (total_c - cell_hits) << "/" << total_c << ")" << std::endl;
    std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "% (" << (total_s - sudoku_hits) << "/" << total_s << ")" << std::endl;

    if(!conf.mixed && (zero_errors || dbn_errors)){
        auto tot = zero_errors + dbn_errors;
        std::cout << "Zero errors: " << 100.0 * zero_errors / tot << "% (" << zero_errors << "/" << tot << ")" << std::endl;
        std::cout << "DBN errors: " << 100.0 * dbn_errors / tot << "% (" << dbn_errors << "/" << tot << ")" << std::endl;
    }

    return 0;
}

int command_time(const config& conf){
    auto dbn = std::make_unique<dbn_t>();

    std::ifstream is("dbn.dat", std::ofstream::binary);
    dbn->load(is);

    {
        //1. Image loading

        std::vector<double> il_sum;

        for(auto& image_source_path : conf.files){
            open_image(image_source_path);
        }

        for(auto& image_source_path : conf.files){
            cpp::stop_watch<std::chrono::microseconds> il_watch;

            open_image(image_source_path);

            il_sum.push_back(il_watch.elapsed());
        }

        std::cout << "Image loading: " << std::endl;
        std::cout << "\tmin: " << min(il_sum) << std::endl;
        std::cout << "\tmax: " << max(il_sum) << std::endl;
        std::cout << "\tmean: " << mean(il_sum) << std::endl;
        std::cout << "\tmedian: " << median(il_sum) << std::endl;
    }

    {
        //2. Line detection

        std::vector<double> ld_sum;

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            detect_lines(source_image, dest_image);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);

            cpp::stop_watch<std::chrono::microseconds> ld_watch;

            auto dest_image = source_image.clone();
            detect_lines(source_image, dest_image);

            ld_sum.push_back(ld_watch.elapsed());
        }

        std::cout << "Line Detection: " << std::endl;
        std::cout << "\tmin: " << min(ld_sum) << std::endl;
        std::cout << "\tmax: " << max(ld_sum) << std::endl;
        std::cout << "\tmean: " << mean(ld_sum) << std::endl;
        std::cout << "\tmedian: " << median(ld_sum) << std::endl;
    }

    {
        //2. Grid detection

        std::vector<double> gd_sum;

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            detect_grid(source_image, dest_image, lines);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);

            cpp::stop_watch<std::chrono::microseconds> gd_watch;

            detect_grid(source_image, dest_image, lines);

            gd_sum.push_back(gd_watch.elapsed());
        }

        std::cout << "Grid Detection: " << std::endl;
        std::cout << "\tmin: " << min(gd_sum) << std::endl;
        std::cout << "\tmax: " << max(gd_sum) << std::endl;
        std::cout << "\tmean: " << mean(gd_sum) << std::endl;
        std::cout << "\tmedian: " << median(gd_sum) << std::endl;
    }

    {
        //3. Digit Detection

        std::vector<double> dd_sum;

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);
            split(source_image, dest_image, cells, lines);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);

            cpp::stop_watch<std::chrono::microseconds> dd_watch;

            split(source_image, dest_image, cells, lines);

            dd_sum.push_back(dd_watch.elapsed());
        }

        std::cout << "Digit Detection: " << std::endl;
        std::cout << "\tmin: " << min(dd_sum) << std::endl;
        std::cout << "\tmax: " << max(dd_sum) << std::endl;
        std::cout << "\tmean: " << mean(dd_sum) << std::endl;
        std::cout << "\tmedian: " << median(dd_sum) << std::endl;
    }

    {
        //4. Digit Recognition

        std::vector<double> dr_sum;

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);
            auto image = split(source_image, dest_image, cells, lines);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i, j);

                    if(cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image(conf));
                        answer = dbn->predict_label(weights)+1;
                    }
                }
            }
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);
            auto image = split(source_image, dest_image, cells, lines);

            cpp::stop_watch<std::chrono::microseconds> dr_watch;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i, j);

                    if(cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image(conf));
                        answer = dbn->predict_label(weights)+1;
                    }
                }
            }

            dr_sum.push_back(dr_watch.elapsed());
        }

        std::cout << "Digit Recognition: " << std::endl;
        std::cout << "\tmin: " << min(dr_sum) << std::endl;
        std::cout << "\tmax: " << max(dr_sum) << std::endl;
        std::cout << "\tmean: " << mean(dr_sum) << std::endl;
        std::cout << "\tmedian: " << median(dr_sum) << std::endl;
    }

    {
        //5. Total

        std::vector<double> tot_sum;

        for(auto& image_source_path : conf.files){
            cpp::stop_watch<std::chrono::microseconds> tot_watch;

            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);
            auto image = split(source_image, dest_image, cells, lines);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i,j);

                    if(cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image(conf));
                        answer = dbn->predict_label(weights)+1;
                    }
                }
            }

            tot_sum.push_back(tot_watch.elapsed());
        }

        std::cout << "Total: " << std::endl;
        std::cout << "\tmin: " << min(tot_sum) << std::endl;
        std::cout << "\tmax: " << max(tot_sum) << std::endl;
        std::cout << "\tmean: " << mean(tot_sum) << std::endl;
        std::cout << "\tmedian: " << median(tot_sum) << std::endl;
    }

    return 0;
}

} //end of anonymous namespace

int main(int argc, char** argv){
    if(argc < 2){
        print_usage();
        return -1;
    }

    auto conf = parse_args(argc, argv);

    conf.gray = conf.mixed && mixed_dbn_t::rbm_type<0>::visible_unit == dll::unit_type::GAUSSIAN;
    conf.big = conf.mixed && std::is_same<mixed_dbn_t, mixed_dbn_pmp_big_t>::value;

    std::cout << "Gray: " << conf.gray << std::endl;;
    std::cout << "Big: " << conf.big << std::endl;;

    if(conf.command == "detect" || conf.command == "detect_save"){
        return command_detect(conf);
    } else if(conf.command == "fill" || conf.command == "fill_save"){
        return command_fill(conf);
    } else if(conf.command == "train"){
        return command_train(conf);
    } else if(conf.command == "recog" || conf.command == "recog_binary"){
        return command_recog(conf);
    } else if(conf.command == "test"){
        return command_test(conf);
    } else if(conf.command == "time"){
        return command_time(conf);
    }

    print_usage();

    return -1;
}
