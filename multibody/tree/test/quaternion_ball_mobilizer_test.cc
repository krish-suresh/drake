#include "drake/multibody/tree/quaternion_ball_mobilizer.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/tree/multibody_tree-inl.h"
#include "drake/multibody/tree/ball_quaternion_joint.h"
#include "drake/multibody/tree/test/mobilizer_tester.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

using Eigen::Quaterniond;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;
using std::make_unique;
using std::unique_ptr;
using systems::Context;

// constexpr double kTolerance = 10 * std::numeric_limits<double>::epsilon();

class QuaternionBallMobilizerTest : public MobilizerTester {
 public:
  void SetUp() override {
    mobilizer_ = &AddJointAndFinalize<BallQuaternionJoint, QuaternionBallMobilizer>(
        std::make_unique<BallQuaternionJoint<double>>(
            "joint0", tree().world_body().body_frame(), body_->body_frame()));
  }

 protected:
  const QuaternionBallMobilizer<double>* mobilizer_{nullptr};
};

TEST_F(QuaternionBallMobilizerTest, CanRotateOrTranslate) {
  EXPECT_TRUE(mobilizer_->can_rotate());
  EXPECT_FALSE(mobilizer_->can_translate());
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
