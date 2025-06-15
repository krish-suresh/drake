#include "drake/multibody/tree/quaternion_ball_mobilizer.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/tree/ball_quaternion_joint.h"
#include "drake/multibody/tree/multibody_tree-inl.h"
#include "drake/multibody/tree/test/mobilizer_tester.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;
using std::make_unique;
using std::unique_ptr;
using systems::Context;

constexpr double kTolerance = 10 * std::numeric_limits<double>::epsilon();

class QuaternionBallMobilizerTest : public MobilizerTester {
 public:
  void SetUp() override {
    mobilizer_ =
        &AddJointAndFinalize<BallQuaternionJoint, QuaternionBallMobilizer>(
            std::make_unique<BallQuaternionJoint<double>>(
                "joint0", tree().world_body().body_frame(),
                body_->body_frame()));
  }

 protected:
  const QuaternionBallMobilizer<double>* mobilizer_{nullptr};
};

TEST_F(QuaternionBallMobilizerTest, CanRotateOrTranslate) {
  EXPECT_TRUE(mobilizer_->can_rotate());
  EXPECT_FALSE(mobilizer_->can_translate());
}

TEST_F(QuaternionBallMobilizerTest, StateAccess) {
  const Quaterniond quaternion_value(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), quaternion_value);
  EXPECT_EQ(mobilizer_->get_quaternion(*context_).coeffs(),
            quaternion_value.coeffs());

  // Set mobilizer orientation using a rotation matrix.
  const RotationMatrixd R_WB(RollPitchYawd(M_PI / 5, -M_PI / 7, M_PI / 3));
  const Quaterniond Q_WB = R_WB.ToQuaternion();
  mobilizer_->SetOrientation(context_.get(), R_WB);
  EXPECT_TRUE(CompareMatrices(mobilizer_->get_quaternion(*context_).coeffs(),
                              Q_WB.coeffs(), kTolerance,
                              MatrixCompareType::relative));
}

TEST_F(QuaternionBallMobilizerTest, ZeroState) {
  // Set an arbitrary "non-zero" state.
  const Quaterniond quaternion_value(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), quaternion_value);
  EXPECT_EQ(mobilizer_->get_quaternion(*context_).coeffs(),
            quaternion_value.coeffs());

  // Set the "zero state" for this mobilizer, which does happen to be that of
  // an identity rigid transform.
  mobilizer_->SetZeroState(*context_, &context_->get_mutable_state());
  const RigidTransformd X_WB(
      mobilizer_->CalcAcrossMobilizerTransform(*context_));
  EXPECT_TRUE(X_WB.IsExactlyIdentity());
}

TEST_F(QuaternionBallMobilizerTest, CalcAcrossMobilizerTransform) {
  const double kTol = 4 * std::numeric_limits<double>::epsilon();
  // Set an arbitrary "non-zero" state.
  const Quaterniond quaternion(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), quaternion);
  const double* q =
      &context_
           ->get_continuous_state_vector()[mobilizer_->position_start_in_q()];
  RigidTransformd X_FM(mobilizer_->CalcAcrossMobilizerTransform(*context_));

  const RigidTransformd X_FM_expected(quaternion, Vector3d::Zero());
  EXPECT_TRUE(X_FM.IsNearlyEqualTo(X_FM_expected, kTol));

  // Now check the fast inline methods.
  RigidTransformd fast_X_FM = mobilizer_->calc_X_FM(q);
  EXPECT_TRUE(fast_X_FM.IsNearlyEqualTo(X_FM, kTol));
  const Quaterniond new_quaternion(
      RollPitchYawd(M_PI / 4, -M_PI / 4, M_PI / 7).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), new_quaternion);
  X_FM = mobilizer_->CalcAcrossMobilizerTransform(*context_);
  mobilizer_->update_X_FM(q, &fast_X_FM);
  EXPECT_TRUE(fast_X_FM.IsNearlyEqualTo(X_FM, kTol));

  TestApplyR_FM(X_FM, *mobilizer_);
  TestPrePostMultiplyByX_FM(X_FM, *mobilizer_);
}

// For an arbitrary state verify that the computed Nplus(q) matrix is the
// left pseudoinverse of N(q).
TEST_F(QuaternionBallMobilizerTest, KinematicMapping) {
  const Quaterniond Q_WB(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), Q_WB);

  ASSERT_EQ(mobilizer_->num_positions(), 4);
  ASSERT_EQ(mobilizer_->num_velocities(), 3);

  // Compute N.
  MatrixX<double> N(4, 3);
  mobilizer_->CalcNMatrix(*context_, &N);

  // Compute Nplus.
  MatrixX<double> Nplus(3, 4);
  mobilizer_->CalcNplusMatrix(*context_, &Nplus);

  // Verify that Nplus is the left pseudoinverse of N.
  MatrixX<double> Nplus_x_N = Nplus * N;

  EXPECT_TRUE(CompareMatrices(Nplus_x_N, MatrixX<double>::Identity(3, 3),
                              kTolerance, MatrixCompareType::relative));

  // Until it is implemented, ensure calculating Ṅ(q,q̇) throws an exception.
  MatrixX<double> NDot(4, 3);
  DRAKE_EXPECT_THROWS_MESSAGE(mobilizer_->CalcNDotMatrix(*context_, &NDot),
                              ".*The function DoCalcNDotMatrix\\(\\) has not "
                              "been implemented for this mobilizer.*");

  // Until it is implemented, ensure calculating Ṅ⁺(q,q̇) throws an exception.
  MatrixX<double> NplusDot(3, 4);
  DRAKE_EXPECT_THROWS_MESSAGE(
      mobilizer_->CalcNplusDotMatrix(*context_, &NplusDot),
      ".*The function DoCalcNplusDotMatrix\\(\\) has not "
      "been implemented for this mobilizer.*");
}

TEST_F(QuaternionBallMobilizerTest, CheckExceptionMessage) {
  const Quaterniond quaternion(0, 0, 0, 0);
  mobilizer_->SetQuaternion(context_.get(), quaternion);

  DRAKE_EXPECT_THROWS_MESSAGE(
      mobilizer_->CalcAcrossMobilizerTransform(*context_),
      "QuaternionToRotationMatrix\\(\\):"
      " All the elements in a quaternion are zero\\.");
}

TEST_F(QuaternionBallMobilizerTest, MapUsesN) {
  // Set an arbitrary "non-zero" state.
  const Quaterniond Q_WB(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), Q_WB);

  EXPECT_FALSE(mobilizer_->is_velocity_equal_to_qdot());

  // Set arbitrary v and MapVelocityToQDot
  const Vector3d v = (Vector3d() << 1.0, 2.0, 3.0).finished();
  VectorX<double> qdot(4);
  mobilizer_->MapVelocityToQDot(*context_, v, &qdot);

  // Compute N.
  MatrixX<double> N(4, 3);
  mobilizer_->CalcNMatrix(*context_, &N);

  // Ensure N(q) is used in `q̇ = N(q)⋅v`
  EXPECT_TRUE(
      CompareMatrices(qdot, N * v, kTolerance, MatrixCompareType::relative));
}

TEST_F(QuaternionBallMobilizerTest, MapUsesNplus) {
  // Set an arbitrary "non-zero" state.
  const Quaterniond Q_WB(
      RollPitchYawd(M_PI / 3, -M_PI / 3, M_PI / 5).ToQuaternion());
  mobilizer_->SetQuaternion(context_.get(), Q_WB);

  // Set arbitrary qdot and MapQDotToVelocity
  VectorX<double> qdot(4);
  qdot << 1.0, 2.0, 3.0, 4.0;

  Vector3d v;
  mobilizer_->MapQDotToVelocity(*context_, qdot, &v);

  // Compute Nplus.
  MatrixX<double> Nplus(3, 4);
  mobilizer_->CalcNplusMatrix(*context_, &Nplus);

  // Ensure N⁺(q) is used in `v = N⁺(q)⋅q̇`
  EXPECT_TRUE(CompareMatrices(v, Nplus * qdot, kTolerance,
                              MatrixCompareType::relative));
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
