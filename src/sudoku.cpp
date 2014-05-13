#include <opencv2/opencv.hpp>

#include <iostream>

#include "dbn/dbn.hpp"
#include "dbn/layer.hpp"
#include "dbn/conf.hpp"
#include "dbn/labels.hpp"
#include "dbn/test.hpp"

#include "detector.hpp"
#include "data.hpp"
#include "image_utils.hpp"

namespace {

vector<double> mat_to_image(const cv::Mat& mat){
    vector<double> image(CELL_SIZE * CELL_SIZE);

    assert(mat.rows == CELL_SIZE);
    assert(mat.cols == CELL_SIZE);

    for(size_t i = 0; i < static_cast<size_t>(mat.rows); ++i){
        for(size_t j = 0; j < static_cast<size_t>(mat.cols); ++j){
            auto value_c = static_cast<std::size_t>(mat.at<uint8_t>(i, j));

            assert(value_c == 0 || value_c == 255);

            image[i * mat.cols + j] = value_c == 0 ? 1.0 : 0.0;
        }
    }

    return image;
}

struct dataset {
    std::vector<vector<double>> training_images;
    std::vector<uint8_t> training_labels;

    std::vector<std::string> source_files;
    std::vector<std::vector<cv::Mat>> source_images;
    std::vector<gt_data> source_data;
};

cv::Mat open_image(const std::string& path){
    auto source_image = cv::imread(path.c_str(), 1);

    if (!source_image.data){
        return source_image;
    }

    if(source_image.rows > 800 || source_image.cols > 800){
        auto factor = 800.0f / std::max(source_image.rows, source_image.cols);

        cv::Mat resized_image;

        cv::resize(source_image, resized_image, cv::Size(), factor, factor, cv::INTER_AREA);

        return resized_image;
    }

    return source_image;
}

dataset get_dataset(int argc, char** argv, bool quiet = false){
    dataset ds;

    for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
        std::string image_source_path(argv[i]);

        if(!quiet){
            std::cout << image_source_path << std::endl;
        }

        auto source_image = open_image(image_source_path);

        if (!source_image.data){
            std::cout << "Invalid source_image" << std::endl;
            continue;
        }

        auto data = read_data(image_source_path);

        cv::Mat dest_image;
        auto mats = detect(source_image, dest_image);

        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                if(data.results[i][j]){
                    ds.training_labels.push_back(data.results[i][j]-1);
                    ds.training_images.emplace_back(mat_to_image(mats[i * 9 + j]));
                }
            }
        }

        ds.source_files.push_back(std::move(image_source_path));
        ds.source_images.push_back(std::move(mats));
        ds.source_data.push_back(std::move(data));
    }

    assert(ds.training_labels.size() == ds.training_images.size());
    assert(ds.source_images.size() == ds.source_data.size());

    return ds;
}

} //end of anonymous namespace

int main(int argc, char** argv ){
    if(argc < 2){
        std::cout << "Usage: sudoku <command> <options>" << std::endl;
        return -1;
    }

    std::string command(argv[1]);

    if(command == "detect" || command == "detect_save"){
        if(argc < 3){
            std::cout << "Usage: sudoku detect <image>..." << std::endl;
            return -1;
        }

        if(argc == 3 && command != "detect_save"){
            std::string image_source_path(argv[2]);

            auto source_image = open_image(image_source_path);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                return -1;
            }

            cv::Mat dest_image;
            detect(source_image, dest_image);

            cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
            cv::imshow("Sudoku Grid", dest_image);

            cv::waitKey(0);
        } else {
            for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
                std::string image_source_path(argv[i]);

                std::cout << image_source_path << std::endl;

                auto source_image = open_image(image_source_path);

                if (!source_image.data){
                    std::cout << "Invalid source_image" << std::endl;
                    continue;
                }

                cv::Mat dest_image;
                detect(source_image, dest_image);

                image_source_path.insert(image_source_path.rfind('.'), ".lines");
                imwrite(image_source_path.c_str(), dest_image);
            }
        }
    } else if(command == "train"){
        auto ds = get_dataset(argc, argv);

        std::cout << "Train with " << ds.source_images.size() << " sudokus" << std::endl;
        std::cout << "Train with " << ds.training_images.size() << " cells" << std::endl;

        auto labels = dbn::make_fake(ds.training_labels);

        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 10, true, true>, CELL_SIZE * CELL_SIZE, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 500>,
            dbn::layer<dbn::conf<true, 10, false, true, true, dbn::Type::EXP>, 500, 9>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();
        dbn->display();

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(ds.training_images, 20);

        std::cout << "Start fine-tuning" << std::endl;
        dbn->fine_tune(ds.training_images, labels, 10, 100);

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);
    } else if(command == "test"){
        auto ds = get_dataset(argc, argv);

        std::cout << "Test with " << ds.source_images.size() << " sudokus" << std::endl;
        std::cout << "Test with " << ds.training_images.size() << " cells" << std::endl;

        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 10, true, true>, CELL_SIZE * CELL_SIZE, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 500>,
            dbn::layer<dbn::conf<true, 10, false, true, true, dbn::Type::EXP>, 500, 9>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::ifstream is("dbn.dat", std::ofstream::binary);
        dbn->load(is);

        auto error_rate = dbn::test_set(dbn, ds.training_images, ds.training_labels, dbn::predictor());

        std::cout << std::endl;
        std::cout << "DBN Error rate (normal): " << 100.0 * error_rate << "%" << std::endl;

        std::size_t sudoku_hits = 0;
        std::size_t cell_hits = 0;
        std::size_t zero_errors = 0;
        std::size_t dbn_errors = 0;

        for(std::size_t i = 0; i < ds.source_images.size(); ++i){
            const auto& image = ds.source_images[i];
            const auto& data = ds.source_data[i];

            std::cout << ds.source_files[i] << std::endl;

            std::size_t local_hits = 0;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell_mat = image[i * 9 + j];

                    auto fill = fill_factor(cell_mat);

                    auto weights = dbn->predict_weights(mat_to_image(cell_mat));
                    if(fill == 1.0f){
                        answer = 0;
                    } else {
                        answer = dbn->predict_final(weights)+1;
                        //std::cout << weights[answer-1] << std::endl;
                        //if(weights[answer-1] < 1e5){
                        //    answer = 0;
                        //}
                    }

                    if(answer == data.results[i][j]){
                        ++local_hits;
                    } else {
                        if(!answer || !data.results[i][j]){
                            ++zero_errors;
                        } else {
                            ++dbn_errors;
                        }

                        std::cout << "ERROR: " << std::endl;
                        std::cout << "\t where: " << i << ":" << j << std::endl;
                        std::cout << "\t answer: " << static_cast<size_t>(answer) << std::endl;
                        std::cout << "\t was: " << static_cast<size_t>(data.results[i][j]) << std::endl;
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

            if(local_hits == 81){
                ++sudoku_hits;
            }

            cell_hits += local_hits;
        }

        auto total_s = static_cast<float>(ds.source_images.size());
        auto total_c = total_s * 81.0f;

        std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "% (" << (total_c - cell_hits) << "/" << total_c << ")" << std::endl;
        std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "% (" << (total_s - sudoku_hits) << "/" << total_s << ")" << std::endl;

        if(zero_errors || dbn_errors){
            auto tot = zero_errors + dbn_errors;
            std::cout << "Zero errors: " << 100.0 * zero_errors / tot << "% (" << zero_errors << "/" << tot << ")" << std::endl;
            std::cout << "DBN errors: " << 100.0 * dbn_errors / tot << "% (" << dbn_errors << "/" << tot << ")" << std::endl;
        }
    } else if(command == "time"){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 10, true, true>, CELL_SIZE * CELL_SIZE, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 10, false, true>, 300, 500>,
            dbn::layer<dbn::conf<true, 10, false, true, true, dbn::Type::EXP>, 500, 9>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        std::ifstream is("dbn.dat", std::ofstream::binary);
        dbn->load(is);

        //1. Image loading

        double il_max = 0.0;
        double il_min = std::numeric_limits<double>::max();
        double il_sum = 0.0;

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            open_image(image_source_path);
        }

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            stop_watch<std::chrono::microseconds> il_watch;
            std::string image_source_path(argv[i]);

            open_image(image_source_path);

            auto il_elapsed = il_watch.elapsed();

            il_max = std::max(il_max, il_elapsed);
            il_min = std::min(il_min, il_elapsed);
            il_sum += il_elapsed;
        }

        std::cout << "Image loading: " << std::endl;
        std::cout << "\tmin: " << il_min << std::endl;
        std::cout << "\tmax: " << il_max << std::endl;
        std::cout << "\taverage: " << (il_sum / (argc - 2)) << std::endl;

        //2. Line detection

        double ld_max = 0.0;
        double ld_min = std::numeric_limits<double>::max();
        double ld_sum = 0.0;

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            detect_lines(source_image, dest_image);
        }

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);

            stop_watch<std::chrono::microseconds> ld_watch;

            auto dest_image = source_image.clone();
            detect_lines(source_image, dest_image);

            auto ld_elapsed = ld_watch.elapsed();

            ld_max = std::max(ld_max, ld_elapsed);
            ld_min = std::min(ld_min, ld_elapsed);
            ld_sum += ld_elapsed;
        }

        std::cout << "Line Detection: " << std::endl;
        std::cout << "\tmin: " << ld_min << std::endl;
        std::cout << "\tmax: " << ld_max << std::endl;
        std::cout << "\taverage: " << (ld_sum / (argc - 2)) << std::endl;

        //2. Grid detection

        double gd_max = 0.0;
        double gd_min = std::numeric_limits<double>::max();
        double gd_sum = 0.0;

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            detect_grid(source_image, dest_image, lines);
        }

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);

            stop_watch<std::chrono::microseconds> gd_watch;

            detect_grid(source_image, dest_image, lines);

            auto gd_elapsed = gd_watch.elapsed();

            gd_max = std::max(gd_max, gd_elapsed);
            gd_min = std::min(gd_min, gd_elapsed);
            gd_sum += gd_elapsed;
        }

        std::cout << "Grid Detection: " << std::endl;
        std::cout << "\tmin: " << gd_min << std::endl;
        std::cout << "\tmax: " << gd_max << std::endl;
        std::cout << "\taverage: " << (gd_sum / (argc - 2)) << std::endl;

        //3. Digit Detection

        double dd_max = 0.0;
        double dd_min = std::numeric_limits<double>::max();
        double dd_sum = 0.0;

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);
            split(source_image, dest_image, cells, lines);
        }

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);
            auto source_image = open_image(image_source_path);
            auto dest_image = source_image.clone();
            auto lines = detect_lines(source_image, dest_image);
            auto cells = detect_grid(source_image, dest_image, lines);

            stop_watch<std::chrono::microseconds> dd_watch;

            split(source_image, dest_image, cells, lines);

            auto dd_elapsed = dd_watch.elapsed();

            dd_max = std::max(dd_max, dd_elapsed);
            dd_min = std::min(dd_min, dd_elapsed);
            dd_sum += dd_elapsed;
        }

        std::cout << "Digit Detection: " << std::endl;
        std::cout << "\tmin: " << dd_min << std::endl;
        std::cout << "\tmax: " << dd_max << std::endl;
        std::cout << "\taverage: " << (dd_sum / (argc - 2)) << std::endl;
    } else {
        std::cout << "Invalid command \"" << command << "\"" << std::endl;
        return -1;
    }

    return 0;
}