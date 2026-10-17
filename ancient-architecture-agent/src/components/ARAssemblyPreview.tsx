import React, { useEffect, useRef, useState } from 'react';
import { Pause, Play, RefreshCw, X } from 'lucide-react';
import * as THREE from 'three';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { applyThreeWoodMaterial } from '../modelMaterials';
import type { DougongAssemblyModel } from '../types/dougongAssembly';

interface ARAssemblyPreviewProps {
  config: DougongAssemblyModel;
  modelName: string;
  onClose: () => void;
}

type RuntimeNode = {
  object: THREE.Object3D;
  assembledPosition: THREE.Vector3;
  explodedPosition: THREE.Vector3;
};

type PointerState = {
  id: number;
  x: number;
  y: number;
};

const BASE_MODEL_SCALE = 0.1;
const MIN_SCALE = 0.55;
const MAX_SCALE = 4.2;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function pointerDistance(first: PointerState, second: PointerState) {
  return Math.hypot(first.x - second.x, first.y - second.y);
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

export default function ARAssemblyPreview({ config, modelName, onClose }: ARAssemblyPreviewProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runtimeNodesRef = useRef<Map<string, RuntimeNode[]>>(new Map());
  const modelPivotRef = useRef<THREE.Group | null>(null);
  const renderFrameRef = useRef<number | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const playbackRunRef = useRef(0);
  const scaleRef = useRef(1);
  const pointersRef = useRef<Map<number, PointerState>>(new Map());
  const gestureRef = useRef({
    startDistance: 0,
    startScale: 1,
    startX: 0,
    startY: 0,
    startRotationX: 0,
    startRotationY: 0,
  });
  const [cameraError, setCameraError] = useState('');
  const [loadError, setLoadError] = useState('');
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(-1);

  useEffect(() => {
    let cancelled = false;

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError('#');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setIsCameraReady(true);
        }
      } catch (error) {
        setCameraError(error instanceof Error ? error.message : '摄像头启动失败');
      }
    }

    void startCamera();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    let disposed = false;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / Math.max(mount.clientHeight, 1), 0.1, 5000);
    camera.position.set(0, 80, 420);
    camera.lookAt(0, 30, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    mount.innerHTML = '';
    mount.appendChild(renderer.domElement);

    const modelPivot = new THREE.Group();
    modelPivot.position.set(0, 0, 0);
    modelPivotRef.current = modelPivot;
    scene.add(modelPivot);

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.7);
    const keyLight = new THREE.DirectionalLight(0xfff3dc, 1.35);
    keyLight.position.set(160, 220, 180);
    const fillLight = new THREE.DirectionalLight(0xd5e8ff, 0.7);
    fillLight.position.set(-180, 120, 100);
    scene.add(ambientLight, keyLight, fillLight);

    const runtimeNodes = runtimeNodesRef.current;
    runtimeNodes.clear();

    function syncExplodedState() {
      config.parts.forEach((part) => {
        const nodes = runtimeNodes.get(part.id) ?? [];
        nodes.forEach((node) => {
          node.object.visible = true;
          node.object.position.copy(node.explodedPosition);
        });
      });
      setStepIndex(-1);
    }

    function startRenderLoop() {
      const render = () => {
        renderFrameRef.current = window.requestAnimationFrame(render);
        renderer.render(scene, camera);
      };
      render();
    }

    function resize() {
      camera.aspect = mount.clientWidth / Math.max(mount.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    }

    window.addEventListener('resize', resize);

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
            root.scale.setScalar(BASE_MODEL_SCALE);
            root.rotation.x = -Math.PI / 2;
            modelPivot.add(root);

            const box = new THREE.Box3().setFromObject(root);
            const center = box.getCenter(new THREE.Vector3());
            root.position.sub(center);

            config.parts.forEach((part) => {
              const nodes: RuntimeNode[] = [];
              root.traverse((child) => {
                if (!part.nodeNames.includes(child.name)) {
                  return;
                }
                const assembledPosition = child.position.clone();
                const explodedPosition = assembledPosition.clone().add(
                  new THREE.Vector3(part.explodeOffset.x, part.explodeOffset.y, part.explodeOffset.z),
                );
                applyPartColor(child, part.color);
                nodes.push({ object: child, assembledPosition, explodedPosition });
              });
              runtimeNodes.set(part.id, nodes);
            });

            syncExplodedState();
            setIsLoaded(true);
            setLoadError('');
            startRenderLoop();
          },
          undefined,
          (error) => {
            if (!disposed) {
              setLoadError(error instanceof Error ? error.message : '模型加载失败');
            }
          },
        );
      },
      undefined,
      (error) => {
        if (!disposed) {
          setLoadError(error instanceof Error ? error.message : '材质加载失败');
        }
      },
    );

    return () => {
      disposed = true;
      playbackRunRef.current += 1;
      if (renderFrameRef.current !== null) {
        window.cancelAnimationFrame(renderFrameRef.current);
        renderFrameRef.current = null;
      }
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current);
        playbackFrameRef.current = null;
      }
      window.removeEventListener('resize', resize);
      renderer.dispose();
      runtimeNodes.clear();
      modelPivotRef.current = null;
      mount.innerHTML = '';
    };
  }, [config]);

  function resetView() {
    playbackRunRef.current += 1;
    if (playbackFrameRef.current !== null) {
      window.cancelAnimationFrame(playbackFrameRef.current);
      playbackFrameRef.current = null;
    }
    const pivot = modelPivotRef.current;
    if (pivot) {
      scaleRef.current = 1;
      pivot.scale.setScalar(1);
      pivot.rotation.set(0, 0, 0);
    }
    runtimeNodesRef.current.forEach((nodes) => {
      nodes.forEach((node) => node.object.position.copy(node.explodedPosition));
    });
    setStepIndex(-1);
    setIsPlaying(false);
  }

  function playAssembly() {
    if (!isLoaded) {
      return;
    }

    playbackRunRef.current += 1;
    const runId = playbackRunRef.current;
    const steps = config.steps;
    setIsPlaying(true);

    runtimeNodesRef.current.forEach((nodes) => {
      nodes.forEach((node) => {
        node.object.visible = true;
        node.object.position.copy(node.explodedPosition);
      });
    });

    const animateNodes = (partIds: string[], progress: number) => {
      partIds.forEach((partId) => {
        const nodes = runtimeNodesRef.current.get(partId) ?? [];
        nodes.forEach((node) => {
          node.object.position.lerpVectors(node.explodedPosition, node.assembledPosition, progress);
        });
      });
    };

    const playStep = (nextStepIndex: number) => {
      if (runId !== playbackRunRef.current) {
        return;
      }

      const step = steps[nextStepIndex];
      if (!step) {
        setStepIndex(steps.length - 1);
        setIsPlaying(false);
        return;
      }

      setStepIndex(nextStepIndex);
      const startedAt = performance.now();
      const tick = (now: number) => {
        if (runId !== playbackRunRef.current) {
          return;
        }

        const progress = Math.min((now - startedAt) / step.durationMs, 1);
        animateNodes(step.partIds, easeInOutCubic(progress));
        if (progress < 1) {
          playbackFrameRef.current = window.requestAnimationFrame(tick);
          return;
        }
        playStep(nextStepIndex + 1);
      };

      playbackFrameRef.current = window.requestAnimationFrame(tick);
    };

    playStep(0);
  }

  function togglePlayback() {
    if (isPlaying) {
      playbackRunRef.current += 1;
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current);
        playbackFrameRef.current = null;
      }
      setIsPlaying(false);
      return;
    }

    playAssembly();
  }

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    pointersRef.current.set(event.pointerId, { id: event.pointerId, x: event.clientX, y: event.clientY });
    const pointers = Array.from(pointersRef.current.values()) as PointerState[];
    const pivot = modelPivotRef.current;
    if (!pivot) {
      return;
    }

    if (pointers.length >= 2) {
      gestureRef.current.startDistance = pointerDistance(pointers[0], pointers[1]);
      gestureRef.current.startScale = scaleRef.current;
    } else {
      gestureRef.current.startX = event.clientX;
      gestureRef.current.startY = event.clientY;
      gestureRef.current.startRotationX = pivot.rotation.x;
      gestureRef.current.startRotationY = pivot.rotation.y;
    }
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!pointersRef.current.has(event.pointerId)) {
      return;
    }
    pointersRef.current.set(event.pointerId, { id: event.pointerId, x: event.clientX, y: event.clientY });
    const pointers = Array.from(pointersRef.current.values()) as PointerState[];
    const pivot = modelPivotRef.current;
    if (!pivot) {
      return;
    }

    if (pointers.length >= 2) {
      const distance = pointerDistance(pointers[0], pointers[1]);
      const nextScale = clamp(
        gestureRef.current.startScale * (distance / Math.max(gestureRef.current.startDistance, 1)),
        MIN_SCALE,
        MAX_SCALE,
      );
      scaleRef.current = nextScale;
      pivot.scale.setScalar(nextScale);
      return;
    }

    const dx = event.clientX - gestureRef.current.startX;
    const dy = event.clientY - gestureRef.current.startY;
    pivot.rotation.y = gestureRef.current.startRotationY + dx * 0.008;
    pivot.rotation.x = clamp(gestureRef.current.startRotationX + dy * 0.005, -0.55, 0.55);
  }

  function handlePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    pointersRef.current.delete(event.pointerId);
    const pointers = Array.from(pointersRef.current.values()) as PointerState[];
    const pivot = modelPivotRef.current;
    if (!pivot) {
      return;
    }

    if (pointers.length === 1) {
      gestureRef.current.startX = pointers[0].x;
      gestureRef.current.startY = pointers[0].y;
      gestureRef.current.startRotationX = pivot.rotation.x;
      gestureRef.current.startRotationY = pivot.rotation.y;
    }
  }

  const activeStep = stepIndex >= 0 ? config.steps[stepIndex] : null;

  return (
    <div className="fixed inset-0 z-[100] bg-black text-white">
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover"
        playsInline
        muted
        autoPlay
      />
      <div
        ref={mountRef}
        className="absolute inset-0 touch-none"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 bg-gradient-to-b from-black/70 to-transparent px-4 pb-10 pt-4">
        <div className="pointer-events-auto flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-widest text-white/55">Camera AR Preview</p>
            <h2 className="mt-1 truncate text-base font-bold">{modelName}</h2>
            <p className="mt-1 text-xs text-white/65">
              {activeStep ? activeStep.label : '双指缩放，单指旋转，模型固定在屏幕中央'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full bg-black/55 p-2 text-white transition-colors hover:bg-black/75"
            aria-label="关闭 AR 预览"
          >
            <X size={20} />
          </button>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 to-transparent px-4 pb-5 pt-12">
        <div className="pointer-events-auto mx-auto flex w-fit items-center gap-2 rounded-full border border-white/15 bg-black/65 p-2 backdrop-blur-md">
          <button
            type="button"
            onClick={resetView}
            className="rounded-full p-3 text-white/75 transition-colors hover:bg-white/10 hover:text-white"
            aria-label="重置 AR 视图"
          >
            <RefreshCw size={20} />
          </button>
          <button
            type="button"
            onClick={togglePlayback}
            disabled={!isLoaded}
            className="rounded-full bg-accent-strong p-3 text-white transition-colors hover:bg-accent-strong/90 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={isPlaying ? '暂停组装动画' : '播放组装动画'}
          >
            {isPlaying ? <Pause size={22} /> : <Play size={22} />}
          </button>
        </div>
      </div>

      {(!isCameraReady || !isLoaded || cameraError || loadError) && (
        <div className="pointer-events-none absolute inset-x-4 top-24 rounded-lg border border-white/15 bg-black/65 px-4 py-3 text-sm shadow-xl backdrop-blur-md">
          {cameraError || loadError || (!isCameraReady ? '正在启动摄像头...' : '正在加载斗拱模型...')}
        </div>
      )}
    </div>
  );
}
