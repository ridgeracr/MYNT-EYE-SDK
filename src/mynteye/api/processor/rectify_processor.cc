// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "mynteye/api/processor/rectify_processor.h"

#include <utility>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mynteye/logger.h"

MYNTEYE_BEGIN_NAMESPACE

cv::Mat RectifyProcessor::rectifyrad(const cv::Mat& R) {
  cv::Mat r_vec;
  cv::Rodrigues(R, r_vec);
  //  pi/180 = x/179 ==> x = 3.1241
  double rad = cv::norm(r_vec);
  if (rad >= 3.1241) {
    cv::Mat r_dir;
    cv::normalize(r_vec, r_dir);
    cv::Mat r = r_dir*(3.1415926 - rad);
    cv::Mat r_r;
    cv::Rodrigues(r, r_r);
    return r_r.clone();
  }
  return R.clone();
}

void RectifyProcessor::stereoRectify(models::CameraPtr leftOdo,
    models::CameraPtr rightOdo, const cv::Mat* K1, const cv::Mat* K2,
    const cv::Mat* D1, const cv::Mat* D2, cv::Size imageSize,
    const cv::Mat* matR, const cv::Mat* matT,
    cv::Mat* _R1, cv::Mat* _R2, cv::Mat* _P1, cv::Mat* _P2, double* T_mul_f,
    double* cx1_min_cx2,
    int flags, double alpha,
    cv::Size newImgSize) {
  
  // Create local copies of matrices
  cv::Mat c_R = *matR;
  cv::Mat c_t = *matT;
  cv::Mat c_K1 = *K1;
  cv::Mat c_K2 = *K2;
  cv::Mat c_D1 = *D1;
  cv::Mat c_D2 = *D2;
  cv::Mat c_R1 = *_R1;
  cv::Mat c_R2 = *_R2;
  cv::Mat c_P1 = *_P1;
  cv::Mat c_P2 = *_P2;

  // Create temporary ROI rectangles
  cv::Rect roi1, roi2;

  // Call OpenCV's stereoRectify with proper ROI parameters
  cv::stereoRectify(c_K1, c_D1, c_K2, c_D2,
      imageSize, c_R, c_t,
      c_R1, c_R2, c_P1, c_P2,
      cv::noArray(), // Q matrix (optional)
      flags,
      alpha,
      newImgSize,
      &roi1, &roi2);  // Pass ROI rectangles instead of doubles

  // Store the values we need from the ROIs
  if (T_mul_f != nullptr) {
    *T_mul_f = static_cast<double>(roi1.width);  // or another appropriate value
  }
  if (cx1_min_cx2 != nullptr) {
    *cx1_min_cx2 = static_cast<double>(roi1.x - roi2.x);  // or another appropriate value
  }

  // Copy results back to output parameters
  c_R1.copyTo(*_R1);
  c_R2.copyTo(*_R2);
  c_P1.copyTo(*_P1);
  c_P2.copyTo(*_P2);
}

// Eigen::Matrix4d RectifyProcessor::loadT(const mynteye::Extrinsics& in) {
  // subEigen
models::Matrix4d RectifyProcessor::loadT(const mynteye::Extrinsics &in) {
  models::Matrix3d R(3);
  R<<
  in.rotation[0][0] << in.rotation[0][1] << in.rotation[0][2] <<
  in.rotation[1][0] << in.rotation[1][1] << in.rotation[1][2] <<
  in.rotation[2][0] << in.rotation[2][1] << in.rotation[2][2];

  double t_x = in.translation[0];
  double t_y = in.translation[1];
  double t_z = in.translation[2];

  models::Quaterniond q;
  q = models::Quaterniond(R);
  
  models::Matrix4d T(4);
  T(3, 3) = 1;
  T.topLeftCorner<3, 3>() = q.toRotationMatrix();
  models::Vector3d t(3, 1);
  t << t_x << t_y << t_z;
  T.topRightCorner<3, 1>() = t;

  return T;
}

void RectifyProcessor::loadCameraMatrix(cv::Mat& K, cv::Mat& D,  // NOLINT
    cv::Size& image_size,  // NOLINT
    struct CameraROSMsgInfo& calib_data) {  // NOLINT
  K = cv::Mat(3, 3, CV_64F, calib_data.K);
  std::size_t d_length = 4;
  D = cv::Mat(1, d_length, CV_64F, calib_data.D);
  image_size = cv::Size(calib_data.width, calib_data.height);
}

struct CameraROSMsgInfo RectifyProcessor::getCalibMatData(
    const mynteye::IntrinsicsEquidistant& in) {
  struct CameraROSMsgInfo calib_mat_data;
  calib_mat_data.distortion_model = "KANNALA_BRANDT";
  calib_mat_data.height = in.height;
  calib_mat_data.width = in.width;

  for (unsigned int i = 0; i < 4; i++) {
    calib_mat_data.D[i] = in.coeffs[i];
  }

  calib_mat_data.K[0] = in.coeffs[4];  // mu
  calib_mat_data.K[4] = in.coeffs[5];  // mv();
  calib_mat_data.K[2] = in.coeffs[6];  // u0();
  calib_mat_data.K[5] = in.coeffs[7];  // v0();
  calib_mat_data.K[8] = 1;
  return calib_mat_data;
}

std::shared_ptr<struct CameraROSMsgInfoPair> RectifyProcessor::stereoRectify(
    models::CameraPtr leftOdo,
    models::CameraPtr rightOdo,
    mynteye::IntrinsicsEquidistant in_left,
    mynteye::IntrinsicsEquidistant in_right, 
    mynteye::Extrinsics ex_right_to_left) {
  models::Matrix4d T = loadT(ex_right_to_left);
  models::Matrix3d R;
  R = T.topLeftCorner<3, 3>();
  models::Vector3d t = T.topRightCorner<3, 1>();

  cv::Mat cv_R(3, 3, CV_64FC1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cv_R.at<double>(i, j) = R(i, j);
    }
  }
  cv::Mat cv_t(3, 1, CV_64FC1);
  for (int i = 0; i < 3; ++i) {
      cv_t.at<double>(i, 0) = t(i, 0);
  }

  cv::Mat K1, D1, K2, D2;
  cv::Size image_size1, image_size2;

  struct CameraROSMsgInfo calib_mat_data_left = getCalibMatData(in_left);
  struct CameraROSMsgInfo calib_mat_data_right = getCalibMatData(in_right);

  loadCameraMatrix(K1, D1, image_size1, calib_mat_data_left);
  loadCameraMatrix(K2, D2, image_size2, calib_mat_data_right);

  cv::Mat R1 = cv::Mat(cv::Size(3, 3), CV_64F);
  cv::Mat R2 = cv::Mat(cv::Size(3, 3), CV_64F);
  cv::Mat P1 = cv::Mat(3, 4, CV_64F);
  cv::Mat P2 = cv::Mat(3, 4, CV_64F);

  // Create CvMat objects from cv::Mat
  cv::Mat c_R = cv_R;
  cv::Mat c_t = cv_t;
  cv::Mat c_K1 = K1;
  cv::Mat c_K2 = K2;
  cv::Mat c_D1 = D1;
  cv::Mat c_D2 = D2;
  cv::Mat c_R1 = R1;
  cv::Mat c_R2 = R2;
  cv::Mat c_P1 = P1;
  cv::Mat c_P2 = P2;

  double T_mul_f;
  double cx1_min_cx2;

  stereoRectify(leftOdo, rightOdo, &c_K1, &c_K2, &c_D1, &c_D2,
      cv::Size(image_size1.width, image_size1.height), &c_R, &c_t, 
      &c_R1, &c_R2, &c_P1, &c_P2, &T_mul_f, &cx1_min_cx2);

#ifdef _DOUTPUT
  std::cout << "K1: " << K1 << std::endl;
  std::cout << "D1: " << D1 << std::endl;
  std::cout << "K2: " << K2 << std::endl;
  std::cout << "D2: " << D2 << std::endl;
  std::cout << "R: " << cv_R << std::endl;
  std::cout << "t: " << cv_t << std::endl;
  std::cout << "R1: " << R1 << std::endl;
  std::cout << "R2: " << R2 << std::endl;
  std::cout << "P1: " << P1 << std::endl;
  std::cout << "P2: " << P2 << std::endl;
#endif
  R1 = rectifyrad(R1);
  R2 = rectifyrad(R2);

  for (std::size_t i = 0; i < 3; i++) {
    for (std::size_t j = 0; j < 4; j++) {
      calib_mat_data_left.P[i*4 + j] = P1.at<double>(i, j);
      calib_mat_data_right.P[i*4 + j] = P2.at<double>(i, j);
    }
  }

  for (std::size_t i = 0; i < 3; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      calib_mat_data_left.R[i*3 + j] = R1.at<double>(i, j);
      calib_mat_data_right.R[i*3 +j] = R2.at<double>(i, j);
    }
  }

  struct CameraROSMsgInfoPair info_pair;
  info_pair.left = calib_mat_data_left;
  info_pair.right = calib_mat_data_right;
  info_pair.T_mul_f = T_mul_f;
  info_pair.cx1_minus_cx2 = cx1_min_cx2;
  for (std::size_t i = 0; i< 3 * 4; i++) {
    info_pair.P[i] = calib_mat_data_left.P[i];
  }

  info_pair.R[0] = ex_right_to_left.rotation[0][0];
  info_pair.R[1] = ex_right_to_left.rotation[0][1];
  info_pair.R[2] = ex_right_to_left.rotation[0][2];
  info_pair.R[3] = ex_right_to_left.rotation[1][0];
  info_pair.R[4] = ex_right_to_left.rotation[1][1];
  info_pair.R[5] = ex_right_to_left.rotation[1][2];
  info_pair.R[6] = ex_right_to_left.rotation[2][0];
  info_pair.R[7] = ex_right_to_left.rotation[2][1];
  info_pair.R[8] = ex_right_to_left.rotation[2][2];

  return std::make_shared<struct CameraROSMsgInfoPair>(info_pair);
}

models::CameraPtr RectifyProcessor::generateCameraFromIntrinsicsEquidistant(
    const mynteye::IntrinsicsEquidistant & in) {
  models::EquidistantCameraPtr camera(
      new models::EquidistantCamera("KANNALA_BRANDT",
                                       in.width,
                                       in.height,
                                       in.coeffs[0],
                                       in.coeffs[1],
                                       in.coeffs[2],
                                       in.coeffs[3],
                                       in.coeffs[4],
                                       in.coeffs[5],
                                       in.coeffs[6],
                                       in.coeffs[7]));
  return camera;
}

void RectifyProcessor::InitParams(
    IntrinsicsEquidistant in_left,
    IntrinsicsEquidistant in_right,
    Extrinsics ex_right_to_left) {
  calib_model = CalibrationModel::KANNALA_BRANDT;
  in_left.ResizeIntrinsics();
  in_right.ResizeIntrinsics();
  in_left_cur = in_left;
  in_right_cur = in_right;
  ex_right_to_left_cur = ex_right_to_left;

  models::CameraPtr camera_odo_ptr_left =
      generateCameraFromIntrinsicsEquidistant(in_left);
  models::CameraPtr camera_odo_ptr_right =
      generateCameraFromIntrinsicsEquidistant(in_right);

  auto calib_info_tmp = stereoRectify(camera_odo_ptr_left,
        camera_odo_ptr_right,
        in_left,
        in_right,
        ex_right_to_left);

  *calib_infos = *calib_info_tmp;
  cv::Mat rect_R_l =
      cv::Mat::eye(3, 3, CV_32F), rect_R_r = cv::Mat::eye(3, 3, CV_32F);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      rect_R_l.at<float>(i, j) = calib_infos->left.R[i*3+j];
      rect_R_r.at<float>(i, j) = calib_infos->right.R[i*3+j];
    }
  }

  double left_f[] =
      {calib_infos->left.P[0], calib_infos->left.P[5]};
  double left_center[] =
      {calib_infos->left.P[2], calib_infos->left.P[6]};
  double right_f[] =
      {calib_infos->right.P[0], calib_infos->right.P[5]};
  double right_center[] =
      {calib_infos->right.P[2], calib_infos->right.P[6]};

  camera_odo_ptr_left->initUndistortRectifyMap(
      map11, map12, left_f[0], left_f[1],
      cv::Size(0, 0), left_center[0],
      left_center[1], rect_R_l);
  camera_odo_ptr_right->initUndistortRectifyMap(
      map21, map22, right_f[0], right_f[1],
      cv::Size(0, 0), right_center[0],
      right_center[1], rect_R_r);
}

const char RectifyProcessor::NAME[] = "RectifyProcessor";

RectifyProcessor::RectifyProcessor(
      std::shared_ptr<IntrinsicsBase> intr_left,
      std::shared_ptr<IntrinsicsBase> intr_right,
      std::shared_ptr<Extrinsics> extr,
      std::int32_t proc_period)
    : Processor(std::move(proc_period)),
      calib_model(CalibrationModel::UNKNOW),
      _alpha(-1) {

  calib_infos = std::make_shared<struct CameraROSMsgInfoPair>();
  InitParams(
    *std::dynamic_pointer_cast<IntrinsicsEquidistant>(intr_left),
    *std::dynamic_pointer_cast<IntrinsicsEquidistant>(intr_right),
    *extr);
}

RectifyProcessor::~RectifyProcessor() {
  VLOG(2) << __func__;
}

std::string RectifyProcessor::Name() {
  return NAME;
}

void RectifyProcessor::ReloadImageParams(
      std::shared_ptr<IntrinsicsBase> intr_left,
      std::shared_ptr<IntrinsicsBase> intr_right,
      std::shared_ptr<Extrinsics> extr) {
  InitParams(
    *std::dynamic_pointer_cast<IntrinsicsEquidistant>(intr_left),
    *std::dynamic_pointer_cast<IntrinsicsEquidistant>(intr_right),
    *extr);
}

Object *RectifyProcessor::OnCreateOutput() {
  return new ObjMat2();
}

bool RectifyProcessor::SetRectifyAlpha(float alpha) {
  _alpha = alpha;
  ReloadImageParams();
  return true;
}

bool RectifyProcessor::OnProcess(
    Object *const in, Object *const out,
    std::shared_ptr<Processor> const parent) {
  MYNTEYE_UNUSED(parent)
  const ObjMat2 *input = Object::Cast<ObjMat2>(in);
  ObjMat2 *output = Object::Cast<ObjMat2>(out);
  cv::remap(input->first, output->first, map11, map12, cv::INTER_LINEAR);
  cv::remap(input->second, output->second, map21, map22, cv::INTER_LINEAR);
  output->first_id = input->first_id;
  output->first_data = input->first_data;
  output->second_id = input->second_id;
  output->second_data = input->second_data;
  return true;
}

MYNTEYE_END_NAMESPACE
