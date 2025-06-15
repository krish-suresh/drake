#include "drake/multibody/tree/ball_quaternion_joint.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/quaternion.h"
#include "drake/multibody/tree/multibody_tree-inl.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace {

const double kTolerance = 1.8 * std::numeric_limits<double>::epsilon();

using Eigen::Vector3d;
using math::RigidTransformd;
using math::RotationMatrixd;
using systems::Context;

using Vector4d = Vector<double, 4>;

constexpr double kPositionLowerLimit = -1.0;
constexpr double kPositionUpperLimit = 1.5;
constexpr double kVelocityLowerLimit = -1.1;
constexpr double kVelocityUpperLimit = 1.6;
constexpr double kAccelerationLowerLimit = -1.2;
constexpr double kAccelerationUpperLimit = 1.7;
constexpr double kAngularDamping = 3;
constexpr double kPositionNonZeroDefault =
    (kPositionLowerLimit + kPositionUpperLimit) / 2;

class BallQuaternionJointTest : public ::testing::Test {
 public:
  // Creates a MultibodyTree model of a single free body.
  void SetUp() override {
    // Spatial inertia for adding bodies. The actual value is not important for
    // these tests and therefore we do not initialize it.
    const auto M_B = SpatialInertia<double>::NaN();

    // Create an empty model.
    auto model = std::make_unique<internal::MultibodyTree<double>>();

    // Add a body so we can add a joint between world and body:
    body_ = &model->AddRigidBody("Body", M_B);

    // Add a quaternion floating joint between the world and body:
    joint_ = &model->AddJoint<BallQuaternionJoint>(
        "Joint", model->world_body(), std::nullopt, *body_, std::nullopt,
        kAngularDamping);
    mutable_joint_ = dynamic_cast<BallQuaternionJoint<double>*>(
        &model->get_mutable_joint(joint_->index()));
    DRAKE_DEMAND(mutable_joint_ != nullptr);
    mutable_joint_->set_position_limits(
        Vector4d::Constant(kPositionLowerLimit),
        Vector4d::Constant(kPositionUpperLimit));
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

TEST_F(BallQuaternionJointTest, Type) {
  const Joint<double>& base = *joint_;
  EXPECT_EQ(base.type_name(), BallQuaternionJoint<double>::kTypeName);
}

// Verify the expected number of dofs.
TEST_F(BallQuaternionJointTest, NumDOFs) {
  EXPECT_EQ(tree().num_positions(), 4);
  EXPECT_EQ(tree().num_velocities(), 3);
  EXPECT_EQ(joint_->num_positions(), 4);
  EXPECT_EQ(joint_->num_velocities(), 3);
  EXPECT_EQ(joint_->position_start(), 0);
  EXPECT_EQ(joint_->velocity_start(), 0);
}

TEST_F(BallQuaternionJointTest, GetJointLimits) {
  EXPECT_EQ(joint_->position_lower_limits().size(), 4);
  EXPECT_EQ(joint_->position_upper_limits().size(), 4);
  EXPECT_EQ(joint_->velocity_lower_limits().size(), 3);
  EXPECT_EQ(joint_->velocity_upper_limits().size(), 3);
  EXPECT_EQ(joint_->acceleration_lower_limits().size(), 3);
  EXPECT_EQ(joint_->acceleration_upper_limits().size(), 3);

  EXPECT_EQ(joint_->position_lower_limits(),
            (Vector4d::Constant(kPositionLowerLimit)));
  EXPECT_EQ(joint_->position_upper_limits(),
            (Vector4d::Constant(kPositionUpperLimit)));
  EXPECT_EQ(joint_->velocity_lower_limits(),
            Vector3d::Constant(kVelocityLowerLimit));
  EXPECT_EQ(joint_->velocity_upper_limits(),
            Vector3d::Constant(kVelocityUpperLimit));
  EXPECT_EQ(joint_->acceleration_lower_limits(),
            Vector3d::Constant(kAccelerationLowerLimit));
  EXPECT_EQ(joint_->acceleration_upper_limits(),
            Vector3d::Constant(kAccelerationUpperLimit));
}

TEST_F(BallQuaternionJointTest, Damping) {
  EXPECT_EQ(joint_->default_damping(), kAngularDamping);
  EXPECT_EQ(joint_->default_damping_vector(),
            (Vector3d() << kAngularDamping, kAngularDamping, kAngularDamping)
                .finished());
}

// // Context-dependent value access.
TEST_F(BallQuaternionJointTest, ContextDependentAccess) {
  const Vector3d angular_velocity(0.5, 0.5, 0.5);
  Quaternion<double> quaternion_A(1., 2., 3., 4.);
  Quaternion<double> quaternion_B(5., 6., 7., 8.);
  quaternion_A.normalize();
  quaternion_B.normalize();
  const RigidTransformd transform_A(quaternion_A, Vector3d::Zero());
  const RotationMatrixd rotation_matrix_B(quaternion_B);

  // Test configuration (orientation and translation).
  joint_->SetQuaternion(context_.get(), quaternion_A);
  EXPECT_EQ(joint_->get_quaternion(*context_).coeffs(), quaternion_A.coeffs());

  joint_->SetOrientation(context_.get(), rotation_matrix_B);
  EXPECT_TRUE(math::AreQuaternionsEqualForOrientation(
      joint_->get_quaternion(*context_), quaternion_B, kTolerance));

  joint_->SetOrientation(context_.get(), RotationMatrixd::Identity());
  // Expect roundoff error in converting the quaternion to a rotation matrix.
  // EXPECT_TRUE(
  //     joint_->GetPose(*context_).IsNearlyEqualTo(transform_A, kTolerance));

  // Angular velocity access:
  joint_->set_angular_velocity(context_.get(), angular_velocity);
  EXPECT_EQ(joint_->get_angular_velocity(*context_), angular_velocity);

  // Joint locking.
  joint_->Lock(context_.get());
  EXPECT_EQ(joint_->get_angular_velocity(*context_), Vector3d::Zero());

  // Damping.
  const Vector3d damping = Vector3d::Ones() * kAngularDamping;
  const Vector3d different_damping = Vector3d::Ones() * .15;
  EXPECT_EQ(joint_->GetDampingVector(*context_), damping);
  EXPECT_NO_THROW(joint_->SetDampingVector(context_.get(), different_damping));
  EXPECT_EQ(joint_->GetDampingVector(*context_), different_damping);

  // Expect to throw on invalid damping values.
  EXPECT_THROW(joint_->SetDampingVector(context_.get(), Vector3d::Constant(-1)),
               std::exception);
}

// Tests API to apply torques to joint.
TEST_F(BallQuaternionJointTest, AddInOneForce) {
  const double some_value = M_PI_2;
  MultibodyForces<double> forces(tree());

  // Since adding forces to individual degrees of freedom of this joint does
  // not make physical sense, this method should throw.
  EXPECT_THROW(joint_->AddInOneForce(*context_, 0, some_value, &forces),
               std::exception);
}

// Tests API to add in damping forces.
TEST_F(BallQuaternionJointTest, AddInDampingForces) {
  const Vector3d angular_velocity(0.1, 0.2, 0.3);
  const double angular_damping = 3 * kAngularDamping;

  const Vector3d damping_forces_expected = -angular_damping * angular_velocity;

  joint_->set_angular_velocity(context_.get(), angular_velocity);
  joint_->SetDampingVector(context_.get(), Vector3d::Ones() * angular_damping);

  MultibodyForces<double> forces(tree());
  joint_->AddInDamping(*context_, &forces);
  EXPECT_EQ(forces.generalized_forces(), damping_forces_expected);
}

TEST_F(BallQuaternionJointTest, Clone) {
  auto model_clone = tree().CloneToScalar<AutoDiffXd>();
  const auto& joint_clone =
      dynamic_cast<const BallQuaternionJoint<AutoDiffXd>&>(
          model_clone->get_variant(*joint_));

  EXPECT_EQ(joint_clone.name(), joint_->name());
  EXPECT_EQ(joint_clone.frame_on_parent().index(),
            joint_->frame_on_parent().index());
  EXPECT_EQ(joint_clone.frame_on_child().index(),
            joint_->frame_on_child().index());
  EXPECT_EQ(joint_clone.position_lower_limits(),
            joint_->position_lower_limits());
  EXPECT_EQ(joint_clone.position_upper_limits(),
            joint_->position_upper_limits());
  EXPECT_EQ(joint_clone.velocity_lower_limits(),
            joint_->velocity_lower_limits());
  EXPECT_EQ(joint_clone.velocity_upper_limits(),
            joint_->velocity_upper_limits());
  EXPECT_EQ(joint_clone.acceleration_lower_limits(),
            joint_->acceleration_lower_limits());
  EXPECT_EQ(joint_clone.acceleration_upper_limits(),
            joint_->acceleration_upper_limits());
  EXPECT_EQ(joint_clone.default_damping(), joint_->default_damping());
  EXPECT_EQ(joint_clone.get_default_quaternion().coeffs(),
            joint_->get_default_quaternion().coeffs());
}

TEST_F(BallQuaternionJointTest, SetVelocityAndAccelerationLimits) {
  const double new_lower = -0.2;
  const double new_upper = 0.2;
  // Check for velocity limits.
  mutable_joint_->set_velocity_limits(Vector3d::Constant(new_lower),
                                      Vector3d::Constant(new_upper));
  EXPECT_EQ(joint_->velocity_lower_limits(), Vector3d::Constant(new_lower));
  EXPECT_EQ(joint_->velocity_upper_limits(), Vector3d::Constant(new_upper));
  // Does not match num_velocities().
  EXPECT_THROW(mutable_joint_->set_velocity_limits(VectorX<double>(3),
                                                   VectorX<double>()),
               std::exception);
  EXPECT_THROW(mutable_joint_->set_velocity_limits(VectorX<double>(),
                                                   VectorX<double>(3)),
               std::exception);
  // Lower limit is larger than upper limit.
  EXPECT_THROW(mutable_joint_->set_velocity_limits(Vector3d::Constant(2),
                                                   Vector3d::Constant(0)),
               std::exception);

  // Check for acceleration limits.
  mutable_joint_->set_acceleration_limits(Vector3d::Constant(new_lower),
                                          Vector3d::Constant(new_upper));
  EXPECT_EQ(joint_->acceleration_lower_limits(), Vector3d::Constant(new_lower));
  EXPECT_EQ(joint_->acceleration_upper_limits(), Vector3d::Constant(new_upper));
  // Does not match num_velocities().
  EXPECT_THROW(mutable_joint_->set_acceleration_limits(VectorX<double>(3),
                                                       VectorX<double>()),
               std::exception);
  EXPECT_THROW(mutable_joint_->set_acceleration_limits(VectorX<double>(),
                                                       VectorX<double>(3)),
               std::exception);
  // Lower limit is larger than upper limit.
  EXPECT_THROW(mutable_joint_->set_acceleration_limits(Vector6d::Constant(2),
                                                       Vector6d::Constant(0)),
               std::exception);
}

TEST_F(BallQuaternionJointTest, CanRotateOrTranslate) {
  EXPECT_TRUE(joint_->can_rotate());
  EXPECT_FALSE(joint_->can_translate());
}

TEST_F(BallQuaternionJointTest, NameSuffix) {
  EXPECT_EQ(joint_->position_suffix(0), "qw");
  EXPECT_EQ(joint_->position_suffix(1), "qx");
  EXPECT_EQ(joint_->position_suffix(2), "qy");
  EXPECT_EQ(joint_->position_suffix(3), "qz");
  EXPECT_EQ(joint_->velocity_suffix(0), "wx");
  EXPECT_EQ(joint_->velocity_suffix(1), "wy");
  EXPECT_EQ(joint_->velocity_suffix(2), "wz");
}

TEST_F(BallQuaternionJointTest, DefaultAngles) {
  const Vector4d lower_limit_angles = Vector4d::Constant(kPositionLowerLimit);
  const Vector4d upper_limit_angles = Vector4d::Constant(kPositionUpperLimit);

  const Vector4d default_angles = Vector4d::Identity();

  const Vector4d new_default_angles =
      Vector4d::Constant(kPositionNonZeroDefault);

  const Vector4d out_of_bounds_low_angles =
      lower_limit_angles - Vector4d::Constant(1);
  const Vector4d out_of_bounds_high_angles =
      upper_limit_angles + Vector4d::Constant(1);

  // Constructor should set the default angle to Vector3d::Zero()
  EXPECT_EQ(joint_->default_positions(), default_angles);

  // Setting a new default angle should propagate so that `get_default_angle()`
  // remains correct.
  mutable_joint_->set_default_positions(new_default_angles);
  EXPECT_EQ(joint_->default_positions(), new_default_angles);

  // Setting the default angle out of the bounds of the position limits
  // should NOT throw an exception
  EXPECT_NO_THROW(
      mutable_joint_->set_default_positions(out_of_bounds_low_angles));
  EXPECT_NO_THROW(
      mutable_joint_->set_default_positions(out_of_bounds_high_angles));

  // Try setting the joint with the wrong number of angles. Be somewhat picky
  // about the resulting message.
  const Vector3d bad_angles = Vector3d::Identity();
  DRAKE_EXPECT_THROWS_MESSAGE(mutable_joint_->set_default_positions(bad_angles),
                              ".*set_default_positions.*positions.*input.*3.*"
                              "not.*DefaultModelInstance::Joint.*4.*");
}

GTEST_TEST(QuaternionFloatingJointNoTreeTest, DefaultAnglesErrorNoTree) {
  RigidBody<double> a{"a"};
  RigidBody<double> b{"b"};
  BallQuaternionJoint<double> dut("dut", a.body_frame(), b.body_frame());
  // Try setting the joint with the wrong number of angles, and no parent
  // tree. Be somewhat picky about the resulting message.
  const Vector3d bad_angles = Vector3d::Identity();
  DRAKE_EXPECT_THROWS_MESSAGE(
      dut.set_default_positions(bad_angles),
      ".*set_default_positions.*positions.*input.*3.*not.*'dut'.*4.*");
}

// TEST_F(BallQuaternionJointTest, RandomState) {
//   RandomGenerator generator;
//   std::uniform_real_distribution<symbolic::Expression> uniform;

//   // Default behavior is to set to zero.
//   tree().SetRandomState(*context_, &context_->get_mutable_state(),
//   &generator); EXPECT_TRUE(joint_->GetPose(*context_).IsExactlyIdentity());

//   // Set the position distribution to arbitrary values.
//   Eigen::Matrix<symbolic::Expression, 3, 1> position_distribution;
//   for (int i = 0; i < 3; i++) {
//     position_distribution[i] = uniform(generator) + i + 1.0;
//   }

//   mutable_joint_->set_random_quaternion_distribution(
//       math::UniformlyRandomQuaternion<symbolic::Expression>(&generator));
//   mutable_joint_->set_random_translation_distribution(position_distribution);
//   tree().SetRandomState(*context_, &context_->get_mutable_state(),
//   &generator);
//   // We expect arbitrary non-zero values for the random state.
//   EXPECT_FALSE(joint_->GetPose(*context_).IsExactlyIdentity());

//   // Set position and quaternion distributions back to 0.
//   mutable_joint_->set_random_quaternion_distribution(
//       Eigen::Quaternion<symbolic::Expression>::Identity());
//   mutable_joint_->set_random_translation_distribution(
//       Eigen::Matrix<symbolic::Expression, 3, 1>::Zero());
//   tree().SetRandomState(*context_, &context_->get_mutable_state(),
//   &generator);
//   // We expect zero values for pose.
//   EXPECT_TRUE(joint_->GetPose(*context_).IsExactlyIdentity());

//   // Set the quaternion distribution using built in uniform sampling.
//   mutable_joint_->set_random_quaternion_distribution_to_uniform();
//   tree().SetRandomState(*context_, &context_->get_mutable_state(),
//   &generator);
//   // We expect arbitrary non-zero pose.
//   EXPECT_FALSE(joint_->GetPose(*context_).IsExactlyIdentity());
// }

// TODO(jwnimmer-tri) This file is missing tests of the clone-related functions.

}  // namespace
}  // namespace multibody
}  // namespace drake
