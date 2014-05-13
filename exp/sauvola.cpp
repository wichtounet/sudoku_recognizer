//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

struct integral_image {
    size_t* const image;
    const size_t rows;
    const size_t cols;

    integral_image(size_t rows, size_t cols) : image(new size_t[cols * rows]), rows(rows), cols(cols) {
        //Nothing
    }

    ~integral_image(){
        delete[] image;
    }

    size_t space = 0;

    size_t& operator()(int64_t i, int64_t j){
        return image[i * cols + j];
    }
};

void method_5(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat g;
    cv::cvtColor(source_image, g, CV_RGB2GRAY);

    auto cols = g.cols;
    auto rows = g.rows;

    integral_image sI(rows, cols);
    integral_image sIs(rows, cols);

    integral_image I(rows, cols);
    integral_image Is(rows, cols);

    for(int j = 0; j < cols; ++j){
        sI(0,j) = g.at<char>(0,j);
        sIs(0,j) = sI(0,j) * sI(0, j);
    }

    for(int i = 1; i < rows; i++){
        for(int j = 0; j < cols; j++){
            size_t value = g.at<char>(i,j);
            sI(i,j) = sI(i-1,j) + value;
            sIs(i,j) = sIs(i-1,j) + value * value;
        }
    }

    for(int i = 0; i < rows; ++i){
        I(i,0) = sI(i,0);
        Is(i,0) = sIs(i,0);
    }

    for(int i = 0; i < rows; i++){
        for(int j = 1; j < cols; j++){
            I(i, j) = I(i, j-1) + sI(i,j);
            Is(i, j) = Is(i, j-1) + sIs(i,j);
        }
    }

    double k = 0.2;
    int w = 22;
    size_t R = 128;

    dest_image = g.clone();

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; j++){
            auto i1 = std::max(0, i - w/2);
            auto j1 = std::max(0, j - w/2);
            auto i2 = std::min(rows-1,i + w/2);
            auto j2 = std::min(cols-1,j + w/2);
            double area = (i2-i1+1)*(j2-j1+1);

            double diff = 0;
            double sqdiff = 0;

            if(i1 == 0 && j1 == 0){
                diff = I(i2,j2);
                sqdiff = Is(i2,j2);
            } else if(i1 == 0 && j1 != 0){
                diff = I(i2,j2) - I(i2,j1-1);
                sqdiff = Is(i2,j2) - Is(i2,j1-1);
            } else if(i1 != 0 && j1 == 0){
                diff = I(i2,j2) - I(i1-1,j2);
                sqdiff = Is(i2,j2) - Is(i1-1,j2);
            } else {
                double diagsum = I(i2,j2) + I(i1-1, j1-1);
                double idiagsum = I(i2,j1 - 1) + I(i1-1, j2);
                diff = diagsum - idiagsum;
                double sqdiagsum = Is(i2,j2) + Is(i1-1, j1-1);
                double sqidiagsum = Is(i2,j1 - 1) + Is(i1-1, j2);
                sqdiff = sqdiagsum - sqidiagsum;
            }

            double m = static_cast<double>(diff) / area;
            double s = sqrt((sqdiff - diff * diff / area) / (area -1));
            double t = m * (1 + k * (s / R - 1));

            //std::cout << t << std::endl;

            if(g.at<char>(i,j) < t){
                dest_image.at<char>(i,j) = 255;
            } else {
                dest_image.at<char>(i,j) = 0;
            }
        }
    }
}


