import * as THREE from 'three';
import type { ModelViewerElement } from './model-viewer';

export const DOUGONG_WOOD_COLOR = '#A86A3B';
const DOUGONG_WOOD_COLOR_FACTOR: [number, number, number, number] = [0.66, 0.42, 0.23, 1];
const DOUGONG_WOOD_ROUGHNESS = 0.82;

type ModelViewerMaterial = {
  pbrMetallicRoughness?: {
    baseColorTexture?: {
      setTexture?: (texture: unknown | null) => void;
    };
    setBaseColorFactor?: (factor: [number, number, number, number]) => void;
    setMetallicFactor?: (factor: number) => void;
    setRoughnessFactor?: (factor: number) => void;
  };
};

type ModelViewerWithMaterials = ModelViewerElement & {
  model?: {
    materials?: ModelViewerMaterial[];
  };
};

export function applyModelViewerWoodMaterial(viewer: ModelViewerElement | null) {
  const modelViewer = viewer as ModelViewerWithMaterials | null;
  const materials = modelViewer?.model?.materials;
  if (!materials?.length) {
    return;
  }

  materials.forEach((material) => {
    material.pbrMetallicRoughness?.baseColorTexture?.setTexture?.(null);
    material.pbrMetallicRoughness?.setBaseColorFactor?.(DOUGONG_WOOD_COLOR_FACTOR);
    material.pbrMetallicRoughness?.setMetallicFactor?.(0);
    material.pbrMetallicRoughness?.setRoughnessFactor?.(DOUGONG_WOOD_ROUGHNESS);
  });
}

export function bindModelViewerWoodMaterial(viewer: ModelViewerElement | null) {
  if (!viewer) {
    return () => {};
  }

  const apply = () => {
    applyModelViewerWoodMaterial(viewer);
    window.setTimeout(() => applyModelViewerWoodMaterial(viewer), 0);
  };

  viewer.addEventListener('load', apply);
  apply();

  return () => {
    viewer.removeEventListener('load', apply);
  };
}

export function applyThreeWoodMaterial(root: THREE.Object3D) {
  const woodColor = new THREE.Color(DOUGONG_WOOD_COLOR);

  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) {
      return;
    }

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const updatedMaterials = materials.map((material) => {
      if (material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshPhysicalMaterial) {
        const nextMaterial = material.clone();
        nextMaterial.color.copy(woodColor);
        nextMaterial.map = null;
        nextMaterial.emissive.setHex(0x000000);
        nextMaterial.emissiveMap = null;
        nextMaterial.metalness = 0;
        nextMaterial.roughness = DOUGONG_WOOD_ROUGHNESS;
        nextMaterial.needsUpdate = true;
        return nextMaterial;
      }

      return new THREE.MeshStandardMaterial({
        color: woodColor,
        metalness: 0,
        roughness: DOUGONG_WOOD_ROUGHNESS,
      });
    });

    mesh.material = Array.isArray(mesh.material) ? updatedMaterials : updatedMaterials[0];
  });
}
