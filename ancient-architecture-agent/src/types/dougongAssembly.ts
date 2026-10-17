export interface AssemblyVector3 {
  x: number;
  y: number;
  z: number;
}

export interface DougongAssemblyPart {
  id: string;
  name: string;
  nodeNames: string[];
  layer: number;
  explodeOffset: AssemblyVector3;
  color?: string;
  description?: string;
}

export interface DougongAssemblyStep {
  id: string;
  label: string;
  partIds: string[];
  order: number;
  durationMs: number;
}

export interface DougongAssemblyModel {
  modelId: string;
  sourceAsset: string;
  parts: DougongAssemblyPart[];
  steps: DougongAssemblyStep[];
}
