#!/usr/bin/env bash
# PREREQ — cluster-level install, run once per cluster before 01/02/03.
#
#   ./prereq-install.sh
#
# Idempotent: every step is `helm upgrade --install` / `kubectl apply`, so
# re-running is safe. Versions are pinned to what was verified on the Nscale
# ray-demo cluster (2x g.192.b200.8 bare metal + 3 control-plane VMs).
set -euo pipefail
cd "$(dirname "$0")"

# 1. Namespace + ib_umad.
#    Order matters: without ib_umad the NVIDIA driver container's fabric manager
#    never reaches the NVLink subnet manager, fabric state sticks at "In Progress",
#    every CUDA call returns "system not yet initialized", and nvidia.com/gpu is
#    never advertised. The driver container also only sees /dev/infiniband/umad*
#    if they existed when it started — hence this goes before gpu-operator.
kubectl apply -f 00-namespace.yaml
kubectl apply -f prereq-node-prep.yaml
kubectl -n v4-flash-demo rollout status ds/ib-umad-loader --timeout=3m

# 2. GPU stack. Driver 580.126.20 is the chart default and is CUDA 13 capable,
#    which the deepseekv4 image needs.
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
helm repo update nvidia >/dev/null
helm upgrade --install gpu-operator nvidia/gpu-operator --version v26.3.3 \
  --namespace gpu-operator --create-namespace --wait --timeout 20m

# 3. KubeRay.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null
helm repo update kuberay >/dev/null
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.6.2 \
  --namespace kuberay --create-namespace --wait --timeout 5m

# 4. The RWX StorageClass `shared-nfs`.
#    This cluster's only StorageClass is cinder (RWO block), and cinder volumes
#    never attach to the bare-metal B200 nodes — ControllerPublishVolume times
#    out. They do attach to the control-plane VMs, so the NFS server lives there
#    and re-exports a 300 Gi cinder volume as RWX. Those VMs are 4 CPU / 16 GB
#    with ~2.5 GB free, so the requests stay deliberately small.
helm repo add nfs-ganesha https://kubernetes-sigs.github.io/nfs-ganesha-server-and-external-provisioner/ >/dev/null
helm repo update nfs-ganesha >/dev/null
helm upgrade --install nfs-server nfs-ganesha/nfs-server-provisioner --version 1.8.0 \
  --namespace nfs-storage --create-namespace \
  --set persistence.enabled=true \
  --set persistence.storageClass=cinder \
  --set persistence.size=300Gi \
  --set storageClass.name=shared-nfs \
  --set 'storageClass.mountOptions={vers=4.1,hard,timeo=600,retrans=2,nconnect=8}' \
  --set 'tolerations[0].key=node-role.kubernetes.io/control-plane' \
  --set 'tolerations[0].effect=NoSchedule' \
  --set 'nodeSelector.node-role\.kubernetes\.io/control-plane=' \
  --set resources.requests.cpu=500m \
  --set resources.requests.memory=1Gi \
  --wait --timeout 10m

# 5. Verify. All three must look right before 01/02/03.
echo
echo "--- GPUs per node (want 8 on each B200 node) ---"
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu' || true
echo "--- NVLink fabric (want: Completed / Success) ---"
# `|| true` matters: under `set -o pipefail` a non-matching grep aborts the script, so a
# cluster with a broken fabric would exit here and never print the storage check below —
# exactly when you need all three lines.
kubectl exec -n gpu-operator ds/nvidia-driver-daemonset -- nvidia-smi -q 2>/dev/null \
  | grep -A2 '^    Fabric' | head -6 || true
echo "--- RWX class ---"
kubectl get sc shared-nfs || true
