#pragma once

#include <memory>
#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/tree/frame.h"
#include "drake/multibody/tree/mobilizer_impl.h"
#include "drake/multibody/tree/multibody_tree_topology.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
class QuaternionBallMobilizer final : public MobilizerImpl<T, 4, 3> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuaternionBallMobilizer);
  using MobilizerBase = MobilizerImpl<T, 4, 3>;
  using MobilizerBase::kNq, MobilizerBase::kNv, MobilizerBase::kNx;
  template <typename U>
  using QVector = typename MobilizerBase::template QVector<U>;
  template <typename U>
  using VVector = typename MobilizerBase::template VVector<U>;
  template <typename U>
  using HMatrix = typename MobilizerBase::template HMatrix<U>;

  QuaternionBallMobilizer(const SpanningForest::Mobod& mobod,
                   const Frame<T>& inboard_frame_F,
                   const Frame<T>& outboard_frame_M)
      : MobilizerBase(mobod, inboard_frame_F, outboard_frame_M) {}

  ~QuaternionBallMobilizer() final;

  std::unique_ptr<BodyNode<T>> CreateBodyNode(
      const BodyNode<T>* parent_node, const RigidBody<T>* body,
      const Mobilizer<T>* mobilizer) const final;

  bool has_quaternion_dofs() const final { return true; }

  std::string position_suffix(int position_index_in_mobilizer) const final;
  std::string velocity_suffix(int velocity_index_in_mobilizer) const final;

  bool can_rotate() const final { return true; }
  bool can_translate() const final { return false; }

  Quaternion<T> get_quaternion(const systems::Context<T>& context) const;

//   void set_random_quaternion_distribution(
//       const Eigen::Quaternion<symbolic::Expression>& q_FM);

  const QuaternionBallMobilizer<T>& SetQuaternion(
      systems::Context<T>* context, const Quaternion<T>& q_FM) const;

  const QuaternionBallMobilizer<T>& SetQuaternion(
      const systems::Context<T>& context, const Quaternion<T>& q_FM,
      systems::State<T>* state) const;

  const QuaternionBallMobilizer<T>& SetFromRotationMatrix(
      systems::Context<T>* context, const math::RotationMatrix<T>& R_FM) const {
        const Eigen::Quaternion<T> q_FM = R_FM.ToQuaternion();
        return SetQuaternion(context, q_FM);
    }

  Vector3<T> get_angular_velocity(const systems::Context<T>& context) const;

  const QuaternionBallMobilizer<T>& SetAngularVelocity(systems::Context<T>* context,
                                                const Vector3<T>& w_FM) const;

  const QuaternionBallMobilizer<T>& SetAngularVelocity(
      const systems::Context<T>& context, const Vector3<T>& w_FM,
      systems::State<T>* state) const;

  math::RigidTransform<T> calc_X_FM(const T* q) const {
    DRAKE_ASSERT(q != nullptr);
    return math::RigidTransform<T>(Eigen::Quaternion<T>(q[0], q[1], q[2], q[3]),
                                   Vector3<T>::Zero());
  }

  void update_X_FM(const T* q, math::RigidTransform<T>* X_FM) const {
    DRAKE_ASSERT(q != nullptr && X_FM != nullptr);
    *X_FM = calc_X_FM(q);
  }

  SpatialVelocity<T> calc_V_FM(const T*, const T* v) const {
    const Eigen::Map<const Vector3<T>> w_FM(v);
    return SpatialVelocity<T>(w_FM, Vector3<T>::Zero());
  }

  SpatialAcceleration<T> calc_A_FM(const T*, const T*, const T* vdot) const {
    const Eigen::Map<const Vector3<T>> alpha_FM(vdot);
    return SpatialAcceleration<T>(alpha_FM, Vector3<T>::Zero());
  }

  void calc_tau(const T*, const SpatialForce<T>& F_BMo_F, T* tau) const {
    DRAKE_ASSERT(tau != nullptr);
    Eigen::Map<VVector<T>> tau_as_vector(tau);
    const Vector3<T>& t_BMo_F = F_BMo_F.rotational();
    tau_as_vector = t_BMo_F;
  }

  math::RigidTransform<T> CalcAcrossMobilizerTransform(
      const systems::Context<T>& context) const final;

  SpatialVelocity<T> CalcAcrossMobilizerSpatialVelocity(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& v) const final;

  SpatialAcceleration<T> CalcAcrossMobilizerSpatialAcceleration(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& vdot) const override;

  void ProjectSpatialForce(const systems::Context<T>& context,
                           const SpatialForce<T>& F_Mo_F,
                           Eigen::Ref<VectorX<T>> tau) const override;

  bool is_velocity_equal_to_qdot() const override { return false; }

 protected:
  void DoCalcNMatrix(const systems::Context<T>& context,
                     EigenPtr<MatrixX<T>> N) const final;

  void DoCalcNplusMatrix(const systems::Context<T>& context,
                         EigenPtr<MatrixX<T>> Nplus) const final;

  // Generally, q̈ = Ṅ(q,q̇)⋅v + N(q)⋅v̇. For this mobilizer, Ṅ is not simple.
  void DoCalcNDotMatrix(const systems::Context<T>& context,
                        EigenPtr<MatrixX<T>> Ndot) const final;

  // Generally, v̇ = Ṅ⁺(q,q̇)⋅q̇ + N⁺(q)⋅q̈. For this mobilizer, Ṅ⁺ is not simple.
  void DoCalcNplusDotMatrix(const systems::Context<T>& context,
                            EigenPtr<MatrixX<T>> NplusDot) const final;

  void DoMapVelocityToQDot(const systems::Context<T>& context,
                           const Eigen::Ref<const VectorX<T>>& v,
                           EigenPtr<VectorX<T>> qdot) const final;

  void DoMapQDotToVelocity(const systems::Context<T>& context,
                           const Eigen::Ref<const VectorX<T>>& qdot,
                           EigenPtr<VectorX<T>> v) const final;

  std::unique_ptr<Mobilizer<double>> DoCloneToScalar(
      const MultibodyTree<double>& tree_clone) const override;

  std::unique_ptr<Mobilizer<AutoDiffXd>> DoCloneToScalar(
      const MultibodyTree<AutoDiffXd>& tree_clone) const override;

  std::unique_ptr<Mobilizer<symbolic::Expression>> DoCloneToScalar(
      const MultibodyTree<symbolic::Expression>& tree_clone) const override;

 private:
  // Helper to compute the kinematic map N(q). L ∈ ℝ⁴ˣ³.
  static Eigen::Matrix<T, 4, 3> CalcLMatrix(const Quaternion<T>& q);
  // Helper to compute the kinematic map N(q) from angular velocity to
  // quaternion time derivative for which q̇_WB = N(q)⋅w_WB.
  // With L given by CalcLMatrix we have:
  // N(q) = L(q_FM/2)
  static Eigen::Matrix<T, 4, 3> AngularVelocityToQuaternionRateMatrix(
      const Quaternion<T>& q);

  // Helper to compute the kinematic map N⁺(q) from quaternion time derivative
  // to angular velocity for which w_WB = N⁺(q)⋅q̇_WB.
  // This method can take a non unity quaternion q_tilde such that
  // w_WB = N⁺(q_tilde)⋅q̇_tilde_WB also holds true.
  // With L given by CalcLMatrix we have:
  // N⁺(q) = L(2 q_FM)ᵀ
  static Eigen::Matrix<T, 3, 4> QuaternionRateToAngularVelocityMatrix(
      const Quaternion<T>& q);

  template <typename ToScalar>
  std::unique_ptr<Mobilizer<ToScalar>> TemplatedDoCloneToScalar(
      const MultibodyTree<ToScalar>& tree_clone) const;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::internal::QuaternionBallMobilizer);
