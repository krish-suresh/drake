#include "drake/multibody/tree/ball_rpy_spring.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/text_logging.h"
#include "drake/multibody/tree/ball_rpy_joint.h"
#include "drake/multibody/tree/multibody_tree-inl.h"
#include "drake/multibody/tree/multibody_tree_system.h"
#include "drake/multibody/tree/position_kinematics_cache.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/multibody/tree/velocity_kinematics_cache.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/systems/framework/context.h"

namespace drake {

using systems::Context;

namespace multibody {
namespace internal {
namespace {

constexpr double kTolerance = 10 * std::numeric_limits<double>::epsilon();
constexpr double kDamping = 3;

class SpringTester : public ::testing::Test {
 public:
  void SetUp() override {
    const auto M_B = SpatialInertia<double>::NaN();

    // Create an empty model.
    auto model = std::make_unique<internal::MultibodyTree<double>>();

    // Add some bodies so we can add joints between them:
    body_ = &model->AddRigidBody("Body", M_B);

    // Add a ball rpy joint between the world and body:
    joint_ = &model->AddJoint<BallRpyJoint>("Joint", model->world_body(),
                                            std::nullopt, *body_, std::nullopt,
                                            kDamping);
    // Add spring
    spring_ = &model->AddForceElement<BallRpySpring>(*joint_, nominal_angles_,
                                                     stiffness_);

    // We are done adding modeling elements. Transfer tree to system and get
    // a Context.
    system_ = std::make_unique<MultibodyTreeSystem<double>>(std::move(model));
    context_ = system_->CreateDefaultContext();

    forces_ = std::make_unique<MultibodyForces<double>>(tree());
  }

  void SetJointState(Vector3<double> position, Vector3<double> position_rate) {
    joint_->set_angles(context_.get(), position);
    joint_->set_angular_velocity(context_.get(), position_rate);
  }

  void CalcSpringForces() const {
    forces_->SetZero();
    spring_->CalcAndAddForceContribution(
        *context_, tree().EvalPositionKinematics(*context_),
        tree().EvalVelocityKinematics(*context_), forces_.get());
  }

  const MultibodyTree<double>& tree() const {
    return GetInternalTree(*system_);
  }

 protected:
  std::unique_ptr<MultibodyTreeSystem<double>> system_;
  std::unique_ptr<Context<double>> context_;

  const RigidBody<double>* body_{nullptr};
  const BallRpyJoint<double>* joint_{nullptr};
  const BallRpySpring<double>* spring_{nullptr};
  std::unique_ptr<MultibodyForces<double>> forces_;

  // Parameters of the case.
  const Vector3<double> nominal_angles_{0, 0, 0};  // [m]
  const Vector3<double> stiffness_{1, 1, 1};       // [N/m]
};

TEST_F(SpringTester, ConstructionAndAccessors) {
  EXPECT_EQ(spring_->joint().index(), joint_->index());
  EXPECT_EQ(spring_->stiffness(), stiffness_);
  EXPECT_EQ(spring_->nominal_angles(), nominal_angles_);
}

// Verify the spring applies no forces when the separation equals the
// nominal angle.
TEST_F(SpringTester, NominalAngle) {
  SetJointState(Vector3<double>{0, 0, 0}, Vector3<double>::Zero(3));
  CalcSpringForces();
  const VectorX<double>& generalized_forces = forces_->generalized_forces();
  EXPECT_EQ(generalized_forces, VectorX<double>::Zero(3));
  // Verify the potential energy is zero.
  const double potential_energy = spring_->CalcPotentialEnergy(
      *context_, tree().EvalPositionKinematics(*context_));
  EXPECT_NEAR(potential_energy, 0.0, kTolerance);
}

// Verify forces computation when the spring angle differs from the nominal.
TEST_F(SpringTester, DeltaAngle) {
  Vector3<double> angles{1, 0, 0};
  SetJointState(angles, Vector3<double>::Zero(3));
  CalcSpringForces();
  const VectorX<double>& generalized_forces = forces_->generalized_forces();

  const VectorX<double>& expected_generalized_forces =
      stiffness_.array() * (nominal_angles_ - angles).array();
  // VectorX<double> expected_generalized_forces(2);
  // expected_generalized_forces << expected_torque_magnitude, 0;
  EXPECT_TRUE(CompareMatrices(generalized_forces, expected_generalized_forces,
                              kTolerance, MatrixCompareType::relative));

  // // Verify the value of the potential energy.
  // const double potential_energy_expected =
  //     0.5 * stiffness_ * (angle - nominal_angle_) * (angle - nominal_angle_);
  // const double potential_energy = spring_->CalcPotentialEnergy(
  //     *context_, tree().EvalPositionKinematics(*context_));
  // EXPECT_NEAR(potential_energy, potential_energy_expected, kTolerance);

  // // Since the spring configuration is static, that is velocities are zero,
  // we
  // // expect zero conservative and non-conservative power.
  // const double conservative_power = spring_->CalcConservativePower(
  //     *context_, tree().EvalPositionKinematics(*context_),
  //     tree().EvalVelocityKinematics(*context_));
  // EXPECT_NEAR(conservative_power, 0.0, kTolerance);
  // const double non_conservative_power = spring_->CalcNonConservativePower(
  //     *context_, tree().EvalPositionKinematics(*context_),
  //     tree().EvalVelocityKinematics(*context_));
  // EXPECT_NEAR(non_conservative_power, 0.0, kTolerance);
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
