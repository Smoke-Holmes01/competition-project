import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { applyThreeWoodMaterial } from '../modelMaterials';
import type { DougongAssemblyModel } from '../types/dougongAssembly';

interface DougongAssemblyViewerProps {
  config: DougongAssemblyModel;
  selectedPartId: string | null;
  focusedStepIndex: number;
  isExploded: boolean;
  isPlaying: boolean;
  onPlayingChange?: (playing: boolean) => void;
  onStepChange?: (stepIndex: number) => void;
  onPlaybackComplete?: () => void;
}

type RuntimeNode = {
  object: THREE.Object3D;
  assembledPosition: THREE.Vector3;
  explodedPosition: THREE.Vector3;
};

const CAMERA_DISTANCE = 360;

function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function applyPartColor(object: THREE.Object3D, color?: string) {
  if (!color) {
    return;
  }

  const partColor = new THREE.Color(color);
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) {
      return;
    }

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const updatedMaterials = materials.map((material) => {
      if (material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshPhysicalMaterial) {
        const nextMaterial = material.clone();
        nextMaterial.color.copy(partColor);
        nextMaterial.needsUpdate = true;
        return nextMaterial;
      }

      return new THREE.MeshStandardMaterial({
        color: partColor,
        metalness: 0,
        roughness: 0.82,
      });
    });

    mesh.material = Array.isArray(mesh.material) ? updatedMaterials : updatedMaterials[0];
  });
}

export default function DougongAssemblyViewer({
  config,
  selectedPartId,
  focusedStepIndex,
  isExploded,
  isPlaying,
  onPlayingChange,
  onStepChange,
  onPlaybackComplete,
}: DougongAssemblyViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const runtimeNodesRef = useRef<Map<string, RuntimeNode[]>>(new Map());
  const renderFrameRef = useRef<number | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const playbackRunRef = useRef(0);
  const callbacksRef = useRef({ onPlayingChange, onStepChange, onPlaybackComplete });
  const viewStateRef = useRef({ selectedPartId, focusedStepIndex, isExploded, isPlaying });
  const [isLoaded, setIsLoaded] = useState(false);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    callbacksRef.current = { onPlayingChange, onStepChange, onPlaybackComplete };
  }, [onPlayingChange, onStepChange, onPlaybackComplete]);

  useEffect(() => {
    viewStateRef.current = { selectedPartId, focusedStepIndex, isExploded, isPlaying };
  }, [selectedPartId, focusedStepIndex, isExploded, isPlaying]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    setIsLoaded(false);
    setLoadError('');

    let disposed = false;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#e8dcc7');

    const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / mount.clientHeight, 0.1, 5000);
    camera.position.set(0, 120, CAMERA_DISTANCE);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.innerHTML = '';
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 80, 0);
    controls.update();

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.45);
    const directionalLight = new THREE.DirectionalLight(0xfff4dd, 1.2);
    directionalLight.position.set(160, 220, 180);
    const fillLight = new THREE.DirectionalLight(0xd9c2a7, 0.7);
    fillLight.position.set(-180, 120, -120);
    scene.add(ambientLight, directionalLight, fillLight);

    const grid = new THREE.GridHelper(560, 12, 0xb79b77, 0xd4c2a8);
    grid.position.y = -80;
    scene.add(grid);

    const runtimeNodes = runtimeNodesRef.current;
    runtimeNodes.clear();

    function syncSceneToState(options?: {
      selectedPartId?: string | null;
      focusedStepIndex?: number;
      isExploded?: boolean;
    }) {
      const viewState = viewStateRef.current;
      const nextSelectedPartId = options?.selectedPartId ?? viewState.selectedPartId;
      const nextFocusedStepIndex = options?.focusedStepIndex ?? viewState.focusedStepIndex;
      const nextIsExploded = options?.isExploded ?? viewState.isExploded;
      const visiblePartIds = nextSelectedPartId ? new Set([nextSelectedPartId]) : null;
      const activeStep = nextFocusedStepIndex >= 0 ? config.steps[nextFocusedStepIndex] ?? null : null;
      const assembledIds = activeStep ? new Set(config.steps.slice(0, nextFocusedStepIndex + 1).flatMap((step) => step.partIds)) : new Set<string>();

      config.parts.forEach((part) => {
        const nodes = runtimeNodes.get(part.id) ?? [];
        const isPartVisible = visiblePartIds ? visiblePartIds.has(part.id) : true;
        const targetAssembled = nextFocusedStepIndex >= 0 ? assembledIds.has(part.id) : !nextIsExploded;

        nodes.forEach((node) => {
          node.object.visible = isPartVisible;
          node.object.position.copy(targetAssembled ? node.assembledPosition : node.explodedPosition);
          const isHighlighted = nextSelectedPartId === part.id;
          node.object.traverse((child) => {
            const mesh = child as THREE.Mesh;
            if (!mesh.isMesh) {
              return;
            }
            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            materials.forEach((material) => {
              const next = material as THREE.MeshStandardMaterial;
              next.emissive = isHighlighted ? new THREE.Color('#6c3c17') : new THREE.Color(0x000000);
              next.emissiveIntensity = isHighlighted ? 0.28 : 0;
            });
          });
        });
      });
    }

    const applyInitialState = () => syncSceneToState();

    function cancelPlaybackFrame() {
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current);
        playbackFrameRef.current = null;
      }
    }

    function cancelRenderFrame() {
      if (renderFrameRef.current !== null) {
        window.cancelAnimationFrame(renderFrameRef.current);
        renderFrameRef.current = null;
      }
    }

    function startRenderLoop() {
      const render = () => {
        renderFrameRef.current = window.requestAnimationFrame(render);
        controls.update();
        renderer.render(scene, camera);
      };
      render();
    }

    function playAssemblySequence() {
      playbackRunRef.current += 1;
      const runId = playbackRunRef.current;
      const steps = config.steps;

      config.parts.forEach((part) => {
        const nodes = runtimeNodes.get(part.id) ?? [];
        nodes.forEach((node) => {
          node.object.visible = true;
          node.object.position.copy(node.explodedPosition);
        });
      });

      const animateNodes = (partIds: string[], progress: number) => {
        partIds.forEach((partId) => {
          const nodes = runtimeNodes.get(partId) ?? [];
          nodes.forEach((node) => {
            node.object.position.lerpVectors(node.explodedPosition, node.assembledPosition, progress);
          });
        });
      };

      const playStep = (stepIndex: number) => {
        if (runId !== playbackRunRef.current) {
          return;
        }
        const step = steps[stepIndex];
        if (!step) {
          cancelPlaybackFrame();
          callbacksRef.current.onStepChange?.(steps.length - 1);
          callbacksRef.current.onPlayingChange?.(false);
          callbacksRef.current.onPlaybackComplete?.();
          return;
        }
        callbacksRef.current.onStepChange?.(stepIndex);
        const startedAt = performance.now();
        const tick = (now: number) => {
          if (runId !== playbackRunRef.current) {
            return;
          }
          const elapsed = now - startedAt;
          const progress = Math.min(elapsed / step.durationMs, 1);
          animateNodes(step.partIds, easeInOutCubic(progress));
          if (progress < 1) {
            playbackFrameRef.current = window.requestAnimationFrame(tick);
            return;
          }
          playStep(stepIndex + 1);
        };
        playbackFrameRef.current = window.requestAnimationFrame(tick);
      };

      playStep(0);
    }

    function syncForNonPlayback(options?: {
      selectedPartId?: string | null;
      focusedStepIndex?: number;
      isExploded?: boolean;
    }) {
      if (viewStateRef.current.isPlaying) {
        return;
      }
      syncSceneToState(options);
    }

    const externalSync = {
      applyInitialState,
      syncForNonPlayback,
      playAssemblySequence,
      cancelPlaybackFrame,
      cancelRenderFrame,
      startRenderLoop,
    };
    (mount as HTMLDivElement & { __assemblySync__?: typeof externalSync }).__assemblySync__ = externalSync;

    function fitCamera(root: THREE.Object3D) {
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      camera.position.set(center.x + maxDim * 0.45, center.y + maxDim * 0.3, center.z + maxDim * 1.35);
      controls.target.copy(center);
      controls.update();
    }

    function handleResize() {
      if (!mount) {
        return;
      }
      camera.aspect = mount.clientWidth / Math.max(mount.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    }

    window.addEventListener('resize', handleResize);

    const mtlLoader = new MTLLoader();
    mtlLoader.setResourcePath('/assembly/group_001/');
    mtlLoader.load(
      '/assembly/group_001/model.mtl',
      (materials) => {
        materials.preload();
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load(
          config.sourceAsset,
          (root) => {
            if (disposed) {
              return;
            }
            applyThreeWoodMaterial(root);
            root.scale.setScalar(0.02);
            root.rotation.x = -Math.PI / 2;
            scene.add(root);

            config.parts.forEach((part) => {
              const nodes: RuntimeNode[] = [];
              root.traverse((child) => {
                if (!part.nodeNames.includes(child.name)) {
                  return;
                }
                const assembledPosition = child.position.clone();
                const explodedPosition = assembledPosition.clone().add(new THREE.Vector3(part.explodeOffset.x, part.explodeOffset.y, part.explodeOffset.z));
                applyPartColor(child, part.color);
                nodes.push({ object: child, assembledPosition, explodedPosition });
              });
              runtimeNodes.set(part.id, nodes);
            });

            applyInitialState();
            fitCamera(root);
            setIsLoaded(true);
            setLoadError('');
            startRenderLoop();
          },
          undefined,
          (error) => {
            if (disposed) {
              return;
            }
            setLoadError(error instanceof Error ? error.message : '拆解模型加载失败');
          },
        );
      },
      undefined,
      (error) => {
        if (disposed) {
          return;
        }
        setLoadError(error instanceof Error ? error.message : '材质文件加载失败');
      },
    );

    return () => {
      disposed = true;
      playbackRunRef.current += 1;
      cancelPlaybackFrame();
      cancelRenderFrame();
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      delete (mount as HTMLDivElement & { __assemblySync__?: unknown }).__assemblySync__;
      mount.innerHTML = '';
      runtimeNodes.clear();
    };
  }, [config]);

  useEffect(() => {
    if (!isLoaded || !mountRef.current) {
      return;
    }
    const sync = (mountRef.current as HTMLDivElement & {
      __assemblySync__?: {
        syncForNonPlayback: (options?: {
          selectedPartId?: string | null;
          focusedStepIndex?: number;
          isExploded?: boolean;
        }) => void;
      };
    }).__assemblySync__;
    sync?.syncForNonPlayback({ selectedPartId, focusedStepIndex, isExploded });
  }, [isLoaded, isExploded, isPlaying, focusedStepIndex, selectedPartId]);

  useEffect(() => {
    if (!isLoaded || !isPlaying || !mountRef.current) {
      return;
    }
    const sync = (mountRef.current as HTMLDivElement & {
      __assemblySync__?: {
        playAssemblySequence: () => void;
        cancelPlaybackFrame: () => void;
      };
    }).__assemblySync__;
    sync?.playAssemblySequence();

    return () => {
      playbackRunRef.current += 1;
      sync?.cancelPlaybackFrame();
    };
  }, [isLoaded, isPlaying]);

  return (
    <div className="relative h-full w-full bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.55),_rgba(214,194,164,0.25)_35%,_rgba(171,141,107,0.22)_100%)]">
      <div ref={mountRef} className="h-full w-full" />
      {!isLoaded && !loadError && (
        <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-ink/70">
          正在加载拆解模型...
        </div>
      )}
      {loadError && (
        <div className="absolute inset-x-6 top-6 rounded-lg border border-red-300 bg-white/90 px-4 py-3 text-sm text-red-700 shadow">
          {loadError}
        </div>
      )}
    </div>
  );
}
