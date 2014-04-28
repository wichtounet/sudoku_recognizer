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

            //TODO Rebinarize after resize

            double value_d;
            if(value_c > 200){
                value_d = 0.0;
            } else {
                value_d = 1.0;
            }

            image[i * mat.cols + j] = value_d;
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

dataset get_dataset(int argc, char** argv){
    dataset ds;

    for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
        std::string image_source_path(argv[i]);

        std::cout << image_source_path << std::endl;

        auto source_image = cv::imread(image_source_path.c_str(), 1);

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

    if(command == "detect"){
        if(argc < 3){
            std::cout << "Usage: sudoku detect <image>..." << std::endl;
            return -1;
        }

        if(argc == 3){
            std::string image_source_path(argv[2]);

            auto source_image = cv::imread(image_source_path.c_str(), 1);

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

                auto source_image = cv::imread(image_source_path.c_str(), 1);

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
            dbn::layer<dbn::conf<true, 50, true, true>, CELL_SIZE * CELL_SIZE, 500>,
            dbn::layer<dbn::conf<true, 50, false, true>, 500, 500>,
            dbn::layer<dbn::conf<true, 50, false, true>, 500, 2000>,
            dbn::layer<dbn::conf<true, 50, false, true, true, dbn::Type::EXP>, 2000, 9>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(ds.training_images, 10);

        std::cout << "Start fine-tuning" << std::endl;
        dbn->fine_tune(ds.training_images, labels, 10, 100);

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);

        auto error_rate = dbn::test_set(dbn, ds.training_images, ds.training_labels, dbn::predictor());

        std::cout << std::endl;
        std::cout << "DBN Error rate (normal): " << 100.0 * error_rate << "%" << std::endl;

        std::size_t sudoku_hits = 0;
        std::size_t cell_hits = 0;

        for(std::size_t i = 0; i < ds.source_images.size(); ++i){
            const auto& image = ds.source_images[i];
            const auto& data = ds.source_data[i];

            std::size_t local_hits = 0;

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    uint8_t answer;

                    auto& cell_mat = image[i * 9 + j];

                    auto fill = fill_factor(cell_mat);

                    if(fill == 1.0f){
                        answer = 0;
                    } else {
                        answer = dbn->predict(mat_to_image(cell_mat))+1;
                    }

                    if(answer == data.results[i][j]){
                        ++local_hits;
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

        std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "%" << std::endl;
        std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "%" << std::endl;
    } else if(command == "test"){
        auto ds = get_dataset(argc, argv);

        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true, true>, CELL_SIZE * CELL_SIZE, 500>,
            dbn::layer<dbn::conf<true, 50, false, true>, 500, 500>,
            dbn::layer<dbn::conf<true, 50, false, true>, 500, 2000>,
            dbn::layer<dbn::conf<true, 50, false, true, true, dbn::Type::EXP>, 2000, 9>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::ifstream is("dbn_2000.dat", std::ofstream::binary);
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

                    if(fill == 1.0f){
                        answer = 0;
                    } else {
                        answer = dbn->predict(mat_to_image(cell_mat))+1;
                    }

                    if(answer == data.results[i][j]){
                        ++local_hits;
                    } else {
                        if(!answer || !data.results[i][j]){
                            ++zero_errors;
                        } else {
                            ++dbn_errors;
                        }

                        std::cout << "ERROR: " << static_cast<size_t>(answer) << std::endl;
                        std::cout << "\t was: " << static_cast<size_t>(data.results[i][j]) << std::endl;
                        std::cout << "\t fill_factor=" << fill << std::endl;

                        std::cout << i << ":" << j << std::endl;
//                        std::cout << cell_mat << std::endl;
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

        std::cout << "Cell Error Rate " << 100.0 * (total_c - cell_hits) / total_c << "%" << std::endl;
        std::cout << "Sudoku Error Rate " << 100.0 * (total_s - sudoku_hits) / total_s << "%" << std::endl;

        if(zero_errors || dbn_errors){
            std::cout << "Cell Errors: " << zero_errors + dbn_errors << std::endl;
            std::cout << "Zero errors: " << 100.0 * zero_errors / static_cast<double>(zero_errors + dbn_errors) << std::endl;
            std::cout << "DBN errors: " << 100.0 * dbn_errors / static_cast<double>(zero_errors + dbn_errors) << std::endl;
        }
    } else {
        std::cout << "Invalid command \"" << command << "\"" << std::endl;
        return -1;
    }

    return 0;
}