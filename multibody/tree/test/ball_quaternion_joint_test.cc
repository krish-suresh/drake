// clang-format: off
#include "drake/multibody/tree/multibody_tree-inl.h"
// clang-format: on

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/multibody/tree/ball_quaternion_joint.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace {

class BallQuaternionJointTest : public ::testing::Test {
 public:
  void SetUp() override {
    const auto M_B = SpatialInertia<double>::NaN();

    // Create an empty model.
    auto model = std::make_unique<internal::MultibodyTree<double>>();

    body_ = &model->AddRigidBody("Body", M_B);

    joint_ = &model->AddJoint<BallQuaternionJoint>("Joint", model->world_body(),
                                            std::nullopt, *body_, std::nullopt,
                                            kDamping);
    mutable_joint_ = dynamic_cast<BallQuaternionJoint<double>*>(
        &model->get_mutable_joint(joint_->index()));
    DRAKE_DEMAND(mutable_joint_ != nullptr);
    mutable_joint_->set_position_limits(
        Vector3d::Constant(kPositionLowerLimit),
        Vector3d::Constant(kPositionUpperLimit));
    mutable_joint_->set_velocity_limits(
        Vector3d::Constant(kVelocityLowerLimit),
        Vector3d::Constant(kVelocityUpperLimit));
    mutable_joint_->set_acceleration_limits(
        Vector3d::Constant(kAccelerationLowerLimit),
        Vector3d::Constant(kAccelerationUpperLimit));

    // We are done adding modeling elements. Transfer tree to system and get
    // a Context.
    system_ = std::make_unique<internal::MultibodyTreeSystem<double>>(
        std::move(model), true /* is_discrete */);
    context_ = system_->CreateDefaultContext();
  }

  const internal::MultibodyTree<double>& tree() const {
    return internal::GetInternalTree(*system_);
  }

 protected:
  std::unique_ptr<internal::MultibodyTreeSystem<double>> system_;
  std::unique_ptr<Context<double>> context_;

  const RigidBody<double>* body_{nullptr};
  const BallQuaternionJoint<double>* joint_{nullptr};
  BallQuaternionJoint<double>* mutable_joint_{nullptr};
};

TEST_F(BallRpyJointTest, Type) {
  const Joint<double>& base = *joint_;
  EXPECT_EQ(base.type_name(), BallRpyJoint<double>::kTypeName);
}

// Verify the expected number of dofs.
TEST_F(BallRpyJointTest, NumDOFs) {

}

}  // namespace
}  // namespace multibody
}  // namespace drake
