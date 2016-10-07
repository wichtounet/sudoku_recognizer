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

#include "dll/rbm.hpp"
#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/mp_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include "detector.hpp"
#include "dataset.hpp"
#include "config.hpp"
#include "image_utils.hpp"
#include "utils.hpp"
#include "fill.hpp"

namespace {

const auto dbn_model_file        = "dbn.dat";
const auto cdbn_mixed_model_file = "cdbn_mixed.dat";
const auto dbn_mixed_model_file  = "dbn_mixed.dat";
const auto cdbn_model_file       = "cdbn.dat";

using mixed_dbn_pmp_t = dll::dbn_desc<
    dll::dbn_layers<
        dll::conv_rbm_mp_desc_square<1, CELL_SIZE, 30, 22, 2,
            dll::weight_type<double>,
            dll::momentum,
            dll::weight_decay<dll::decay_type::L2>,
            //dll::visible<dll::unit_type::GAUSSIAN>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>
        >::layer_t,
        dll::conv_rbm_mp_desc_square<30, 11, 30, 6, 2,
            dll::weight_type<double>,
            dll::momentum,
            dll::weight_decay<dll::decay_type::L2>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>>::layer_t/*,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::layer_t*/
    >, dll::batch_size<64>, /*dll::watcher<dll::opencv_dbn_visualizer>, */dll::svm_concatenate/*, dll::svm_scale*/>::dbn_t;

using mixed_dbn_pmp_big_t = dll::dbn_desc<
    dll::dbn_layers<
        dll::conv_rbm_mp_desc_square<1, BIG_CELL_SIZE, 40, 32, 2,
            dll::weight_type<double>,
            dll::momentum,
            dll::weight_decay<dll::decay_type::L2>,
            //dll::visible<dll::unit_type::GAUSSIAN>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>
        >::layer_t,
        dll::conv_rbm_mp_desc_square<40, 16, 40, 10, 2,
            dll::weight_type<double>,
            dll::momentum,
            dll::weight_decay<dll::decay_type::L2>,
            dll::sparsity<dll::sparsity_method::LEE>,
            dll::batch_size<25>>::layer_t/*,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::layer_t*/
    >, dll::batch_size<64>, dll::svm_concatenate/*, dll::svm_scale*/>::dbn_t;

using mixed_dbn_t = mixed_dbn_pmp_t;

using dbn_t =
    dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<CELL_SIZE * CELL_SIZE, 500,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::init_weights
            >::layer_t,
            dll::rbm_desc<500, 1000,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>
            >::layer_t,
            dll::rbm_desc<1000, 9,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::SOFTMAX>
            >::layer_t
        >,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<32>,
        dll::momentum,
        dll::shuffle,
        //dll::verbose,
        dll::weight_decay<dll::decay_type::L2>
    >::dbn_t;

using dbn_mixed_t =
    dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<CELL_SIZE * CELL_SIZE, 300,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::BINARY>,
                dll::init_weights
            >::layer_t,
            dll::rbm_desc<300, 300,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::rbm_desc<300, 9,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::SOFTMAX>
            >::layer_t
        >,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<32>,
        dll::momentum,
        dll::shuffle,
        //dll::verbose,
        dll::weight_decay<dll::decay_type::L2>
    >::dbn_t;

using cdbn_t =
    dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, CELL_SIZE, CELL_SIZE, 4, 28, 28,
                dll::weight_type<float>,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::mp_layer_3d_desc<4, 28, 28, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_rbm_desc<4, 14, 14, 6, 10, 10,
                dll::weight_type<float>,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::mp_layer_3d_desc<6, 10, 10, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::rbm_desc<6 * 5 * 5, 120,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::rbm_desc<120, 9,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::SOFTMAX>
            >::layer_t
        >,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<32>,
        dll::momentum,
        dll::shuffle,
        //dll::verbose,
        dll::weight_decay<dll::decay_type::L2>
    >::dbn_t;

using cdbn_mixed_t =
    dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, CELL_SIZE, CELL_SIZE, 4, 28, 28,
                dll::weight_type<float>,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::mp_layer_3d_desc<4, 28, 28, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_rbm_desc<4, 14, 14, 6, 10, 10,
                dll::weight_type<float>,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::mp_layer_3d_desc<6, 10, 10, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::rbm_desc<6 * 5 * 5, 100,
                dll::momentum,
                dll::shuffle,
                dll::weight_decay<dll::decay_type::L2>,
                dll::batch_size<32>,
                dll::hidden<dll::unit_type::BINARY>
            >::layer_t,
            dll::rbm_desc<100, 9,
                dll::momentum,
                dll::shuffle,
                dll::batch_size<32>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::hidden<dll::unit_type::SOFTMAX>
            >::layer_t
        >,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<32>,
        dll::momentum,
        dll::shuffle,
        //dll::verbose,
        dll::weight_decay<dll::decay_type::L2>
    >::dbn_t;

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
    auto ds = get_dataset(conf);

    std::cout << "Train with " << ds.source_grids.size() << " sudokus" << std::endl;
    std::cout << "Train with " << ds.training_images.size() << " cells" << std::endl;

    if(conf.test){
        std::cout << "Test with " << ds.test_images.size() << " cells" << std::endl;
    }

    bool distort = false;

    if(conf.mixed){
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_mixed_t>();
            dbn->display();

            dbn->layer_get<0>().initial_momentum = 0.9;
            dbn->layer_get<1>().initial_momentum = 0.9;
            dbn->layer_get<1>().learning_rate = 0.05;
            dbn->layer_get<2>().initial_momentum = 0.9;

            dbn->initial_momentum = 0.9;
            dbn->learning_rate    = 0.03;

            dbn->l2_weight_cost = 0.005;
            dbn->goal           = 0.01;

            // Copy the data (for augmentation)
            auto images = ds.training_images_1d();
            auto labels = ds.training_labels;
            auto size = images.size();

            if (distort) {
                images.reserve(size * 5);
                labels.reserve(size * 5);

                for (size_t i = 0; i < size; ++i) {
                    auto& image = images[i];

                    auto copy_left   = image;
                    auto copy_right  = image;
                    auto copy_top    = image;
                    auto copy_bottom = image;

                    copy_left   = 0;
                    copy_right  = 0;
                    copy_top    = 0;
                    copy_bottom = 0;

                    for (size_t x = 0; x < CELL_SIZE; ++x) {
                        for (size_t y = 0; y < CELL_SIZE; ++y) {
                            if (y > 2) {
                                copy_left[x * CELL_SIZE + y] = image[x * CELL_SIZE + y - 2];
                            }

                            if (y < CELL_SIZE - 2) {
                                copy_right[x * CELL_SIZE + y] = image[x * CELL_SIZE + y + 2];
                            }

                            if (x > 2) {
                                copy_top[x * CELL_SIZE + y] = image[(x - 2) * CELL_SIZE + y];
                            }

                            if (x < CELL_SIZE - 2) {
                                copy_bottom[x * CELL_SIZE + y] = image[(x + 2) * CELL_SIZE + y];
                            }
                        }
                    }

                    labels.push_back(labels[i]);
                    labels.push_back(labels[i]);
                    labels.push_back(labels[i]);
                    labels.push_back(labels[i]);

                    images.push_back(std::move(copy_left));
                    images.push_back(std::move(copy_right));
                    images.push_back(std::move(copy_top));
                    images.push_back(std::move(copy_bottom));
                }
            }

            std::cout << "Start pretraining" << std::endl;
            //dbn->pretrain(images, 50);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(images, labels, 200);

            std::cout << "training_error:" << dll::test_set(dbn, images, labels, dll::predictor()) << std::endl;

            std::ofstream os(dbn_mixed_model_file, std::ofstream::binary);
            dbn->store(os);
            std::cout << "store the model in " << dbn_mixed_model_file << std::endl;
        } else {
            auto cdbn = std::make_unique<cdbn_mixed_t>();
            cdbn->display();

            cdbn->layer_get<0>().initial_momentum = 0.9; // C1
            cdbn->layer_get<2>().initial_momentum = 0.9; // C2
            cdbn->layer_get<4>().initial_momentum = 0.9; // F1
            cdbn->layer_get<4>().initial_momentum = 0.9; // F1
            cdbn->layer_get<5>().initial_momentum = 0.9; // F2

            cdbn->layer_get<0>().learning_rate = 1e-3; // C1
            cdbn->layer_get<2>().learning_rate = 1e-4; // C1
            cdbn->layer_get<4>().learning_rate = 1e-3; // R1

            cdbn->initial_momentum = 0.9;
            cdbn->learning_rate = 0.03;

            auto& images = ds.training_images_1d();
            auto& labels = ds.training_labels;

            std::cout << "Start pretraining" << std::endl;
            cdbn->pretrain(images, 100);

            std::cout << "Start fine-tuning" << std::endl;
            cdbn->fine_tune(images, labels, 200);

            std::cout << "training_error:" << dll::test_set(cdbn, images, labels, dll::predictor()) << std::endl;

            std::ofstream os(cdbn_mixed_model_file, std::ofstream::binary);
            cdbn->store(os);
            std::cout << "store the model in " << cdbn_mixed_model_file << std::endl;
        }
    } else {
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_t>();
            dbn->display();

            dbn->layer_get<0>().initial_momentum = 0.9;
            dbn->layer_get<1>().initial_momentum = 0.9;
            dbn->layer_get<2>().initial_momentum = 0.9;

            dbn->initial_momentum = 0.9;
            dbn->learning_rate = 0.01;

            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(ds.training_images_1d(), 50);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(ds.training_images_1d(), ds.training_labels, 100);

            std::cout << "training_error:" << dll::test_set(dbn, ds.training_images_1d(), ds.training_labels, dll::predictor()) << std::endl;

            if(conf.test){
                std::cout << "test_error:" << dll::test_set(dbn, ds.test_images_1d(), ds.test_labels, dll::predictor()) << std::endl;
            }

            std::ofstream os(dbn_model_file, std::ofstream::binary);
            dbn->store(os);
            std::cout << "store the model in " << dbn_model_file << std::endl;
        } else {
            auto cdbn = std::make_unique<cdbn_t>();
            cdbn->display();

            cdbn->layer_get<0>().initial_momentum = 0.9; // C1
            cdbn->layer_get<2>().initial_momentum = 0.9; // C2
            cdbn->layer_get<4>().initial_momentum = 0.9; // F1
            cdbn->layer_get<4>().initial_momentum = 0.9; // F1
            cdbn->layer_get<5>().initial_momentum = 0.9; // F2

            cdbn->layer_get<0>().learning_rate = 1e-3; // C1
            cdbn->layer_get<2>().learning_rate = 1e-4; // C1
            cdbn->layer_get<4>().learning_rate = 1e-3; // R1

            cdbn->initial_momentum = 0.9;
            cdbn->learning_rate = 0.01;

            std::cout << "Start pretraining" << std::endl;
            cdbn->pretrain(ds.training_images_1d(), 100);

            std::cout << "Start fine-tuning" << std::endl;
            cdbn->fine_tune(ds.training_images_1d(), ds.training_labels, 200);

            std::cout << "training_error:" << dll::test_set(cdbn, ds.training_images_1d(), ds.training_labels, dll::predictor()) << std::endl;

            if(conf.test){
                std::cout << "test_error:" << dll::test_set(cdbn, ds.test_images_1d(), ds.test_labels, dll::predictor()) << std::endl;
            }

            std::ofstream os(cdbn_model_file, std::ofstream::binary);
            cdbn->store(os);
            std::cout << "store the model in " << cdbn_model_file << std::endl;
        }
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
                    static constexpr size_t W = mixed_dbn_t::layer_type<0>::NV1;
                    matrix[i][j] = dbn->svm_predict(grid(j, i).image_fast<W>(conf));
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
                        auto weights = dbn->activation_probabilities(cell.image_1d<float>(conf));
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

template<typename Net>
void standard_test_network(const Net& dbn, const config& conf, dataset& ds){
    std::cout << "Start testing in standard mode" << std::endl;

    auto train_error_rate = dll::test_set(dbn, ds.training_images_1d(), ds.training_labels, dll::predictor());
    auto test_error_rate = dll::test_set(dbn, ds.test_images_1d(), ds.test_labels, dll::predictor());
    auto all_error_rate = dll::test_set(dbn, ds.all_images_1d(), ds.all_labels, dll::predictor());

    std::cout << std::endl;
    std::cout << "DBN   Train Error rate (normal): " << 100.0 * train_error_rate << "%" << std::endl;
    std::cout << "DBN    Test Error rate (normal): " << 100.0 * test_error_rate << "%" << std::endl;
    std::cout << "DBN Overall Error rate (normal): " << 100.0 * all_error_rate << "%" << std::endl;

    size_t sudoku_hits = 0;
    size_t cell_hits = 0;
    size_t zero_errors = 0;
    size_t dbn_errors = 0;

    for(std::size_t i = 0; i < ds.source_grids.size(); ++i){
        const auto& grid = ds.source_grids[i];

        std::cout << grid.source_image_path << std::endl;

        std::size_t local_hits = 0;

        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                uint8_t answer;
                auto& cell = grid(j,i);
                auto& cell_mat = cell.mat(conf);
                auto correct = cell.correct();

                auto weights = dbn->activation_probabilities(cell.image_1d<float>(conf));
                if(cell.empty()){
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
                        std::cout << "\t fill_factor: " << fill_factor(cell_mat) << std::endl;

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

    auto total_s = static_cast<double>(ds.source_grids.size());
    auto total_c = total_s * 81.0;

    std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "% (" << (total_c - cell_hits) << "/" << total_c << ")" << std::endl;
    std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "% (" << (total_s - sudoku_hits) << "/" << total_s << ")" << std::endl;

    if(zero_errors || dbn_errors){
        auto tot = zero_errors + dbn_errors;
        std::cout << "Zero errors: " << 100.0 * zero_errors / tot << "% (" << zero_errors << "/" << tot << ")" << std::endl;
        std::cout << "DBN errors: " << 100.0 * dbn_errors / tot << "% (" << dbn_errors << "/" << tot << ")" << std::endl;
    }
}

template<typename Net>
void mixed_test_network(const Net& dbn, const config& conf, dataset& ds){
    std::cout << "Start testing in mixed mode" << std::endl;

    auto train_error_rate = dll::test_set(dbn, ds.training_images_1d(), ds.training_labels, dll::predictor());
    auto test_error_rate = dll::test_set(dbn, ds.test_images_1d(), ds.test_labels, dll::predictor());
    auto all_error_rate = dll::test_set(dbn, ds.all_images_1d(), ds.all_labels, dll::predictor());

    std::cout << std::endl;
    std::cout << "DBN   Train Error rate (normal): " << 100.0 * train_error_rate << "%" << std::endl;
    std::cout << "DBN    Test Error rate (normal): " << 100.0 * test_error_rate << "%" << std::endl;
    std::cout << "DBN Overall Error rate (normal): " << 100.0 * all_error_rate << "%" << std::endl;

    size_t sudoku_hits = 0;
    size_t cell_hits = 0;
    size_t dbn_errors = 0;

    for(std::size_t i = 0; i < ds.source_grids.size(); ++i){
        const auto& grid = ds.source_grids[i];

        std::cout << grid.source_image_path << std::endl;

        std::size_t local_hits = 0;

        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                uint8_t answer;
                auto correct = grid(j,i).correct();

                auto weights = dbn->activation_probabilities(grid(j,i).image_1d<float>(conf));
                answer = dbn->predict_label(weights)+1;

                if(answer == correct){
                    ++local_hits;
                } else {
                    ++dbn_errors;

                    if(!conf.quiet){
                        std::cout << "ERROR: " << std::endl;
                        std::cout << "\t where: " << i << ":" << j << std::endl;
                        std::cout << "\t answer: " << static_cast<size_t>(answer) << std::endl;
                        std::cout << "\t was: " << static_cast<size_t>(correct) << std::endl;

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

    auto total_s = static_cast<double>(ds.source_grids.size());
    auto total_c = total_s * 81.0;
    auto tot = dbn_errors;

    std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "% (" << (total_c - cell_hits) << "/" << total_c << ")" << std::endl;
    std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "% (" << (total_s - sudoku_hits) << "/" << total_s << ")" << std::endl;
    std::cout << "DBN errors: " << 100.0 * dbn_errors / tot << "% (" << dbn_errors << "/" << tot << ")" << std::endl;
}

int command_test(const config& conf){
    auto ds = get_dataset(conf);

    std::cout << "Test with " << ds.source_grids.size() << " sudokus" << std::endl;
    std::cout << "Test with " << ds.all_images.size() << " cells" << std::endl;

    if(conf.mixed){
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_mixed_t>();

            dbn->display();

            std::ifstream is(dbn_mixed_model_file, std::ofstream::binary);
            dbn->load(is);
            std::cout << "Load model from " << dbn_mixed_model_file << std::endl;

            mixed_test_network(dbn, conf, ds);
        } else {
            auto cdbn = std::make_unique<cdbn_mixed_t>();

            cdbn->display();

            std::ifstream is(cdbn_mixed_model_file, std::ofstream::binary);
            cdbn->load(is);
            std::cout << "Load model from " << cdbn_mixed_model_file << std::endl;

            mixed_test_network(cdbn, conf, ds);
        }
    } else {
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_t>();

            dbn->display();

            std::ifstream is(dbn_model_file, std::ofstream::binary);
            dbn->load(is);
            std::cout << "Load model from " << dbn_model_file << std::endl;

            standard_test_network(dbn, conf, ds);
        } else {
            auto cdbn = std::make_unique<cdbn_t>();

            cdbn->display();

            std::ifstream is(cdbn_model_file, std::ofstream::binary);
            cdbn->load(is);
            std::cout << "Load model from " << cdbn_model_file << std::endl;

            standard_test_network(cdbn, conf, ds);
        }
    }

    return 0;
}

template<typename Net>
int time_network(const config& conf, Net& dbn){
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
            detect_lines(source_image, dest_image, conf.mixed);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);

            cpp::stop_watch<std::chrono::microseconds> ld_watch;

            auto dest_image = source_image.clone();
            detect_lines(source_image, dest_image, conf.mixed);

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
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            detect_grid(source_image, dest_image, lines, conf.mixed);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image, conf.mixed);

            cpp::stop_watch<std::chrono::microseconds> gd_watch;

            detect_grid(source_image, dest_image, lines, conf.mixed);

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
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            auto cells = detect_grid(source_image, dest_image, lines, conf.mixed);
            split(source_image, dest_image, cells, lines, conf.mixed);
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            auto cells = detect_grid(source_image, dest_image, lines, conf.mixed);

            cpp::stop_watch<std::chrono::microseconds> dd_watch;

            split(source_image, dest_image, cells, lines, conf.mixed);

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
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            auto cells = detect_grid(source_image, dest_image, lines, conf.mixed);
            auto image = split(source_image, dest_image, cells, lines, conf.mixed);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i, j);

                    if(!conf.mixed && cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image_1d<float>(conf));
                        answer = dbn->predict_label(weights)+1;
                    }

                    cpp_unused(answer);
                }
            }
        }

        for(auto& image_source_path : conf.files){
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            auto cells = detect_grid(source_image, dest_image, lines, conf.mixed);
            auto image = split(source_image, dest_image, cells, lines, conf.mixed);

            cpp::stop_watch<std::chrono::microseconds> dr_watch;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i, j);

                    if(!conf.mixed && cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image_1d<float>(conf));
                        answer = dbn->predict_label(weights)+1;
                    }

                    cpp_unused(answer);
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
            auto lines = detect_lines(source_image, dest_image, conf.mixed);
            auto cells = detect_grid(source_image, dest_image, lines, conf.mixed);
            auto image = split(source_image, dest_image, cells, lines, conf.mixed);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell = image(i,j);

                    if(!conf.mixed && cell.empty()){
                        answer = 0;
                    } else {
                        auto weights = dbn->activation_probabilities(cell.image_1d<float>(conf));
                        answer = dbn->predict_label(weights)+1;
                    }

                    cpp_unused(answer);
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

int command_time(const config& conf){
    if(conf.mixed){
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_mixed_t>();

            std::ifstream is(dbn_mixed_model_file, std::ofstream::binary);
            dbn->load(is);
            std::cout << "Load model from " << dbn_mixed_model_file << std::endl;

            return time_network(conf, dbn);
        } else {
            auto cdbn = std::make_unique<cdbn_mixed_t>();

            std::ifstream is(cdbn_mixed_model_file, std::ofstream::binary);
            cdbn->load(is);
            std::cout << "Load model from " << cdbn_mixed_model_file << std::endl;

            return time_network(conf, cdbn);
        }
    } else {
        if(!conf.conv){
            auto dbn = std::make_unique<dbn_t>();

            std::ifstream is(dbn_model_file, std::ofstream::binary);
            dbn->load(is);
            std::cout << "Load model from " << dbn_model_file << std::endl;

            return time_network(conf, dbn);
        } else {
            auto cdbn = std::make_unique<cdbn_t>();

            std::ifstream is(cdbn_model_file, std::ofstream::binary);
            cdbn->load(is);
            std::cout << "Load model from " << cdbn_model_file << std::endl;

            return time_network(conf, cdbn);
        }
    }
}

} //end of anonymous namespace

int main(int argc, char** argv){
    if(argc < 2){
        print_usage();
        return -1;
    }

    auto conf = parse_args(argc, argv);

    // TODO Maybe review this
    if(conf.mixed){
        if(conf.conv){
            conf.gray = false;
            conf.big = false;
        } else {
            conf.gray = false;
            conf.big = false;
        }
    } else {
        conf.gray = false;
        conf.big  = false;
    }

    if(conf.shuffle){
        std::random_device rd{};
        std::default_random_engine rand_engine{rd()};
        std::shuffle(conf.files.begin(), conf.files.end(), rand_engine);
    }

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
