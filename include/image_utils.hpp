#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

inline float fill_factor(const cv::Mat& mat){
	auto non_zero = cv::countNonZero(mat);
	auto area = mat.cols * mat.rows;
	return (static_cast<float>(non_zero) / area);
}

#endif
