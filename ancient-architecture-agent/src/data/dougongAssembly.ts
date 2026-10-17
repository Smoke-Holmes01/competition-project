import type { DougongAssemblyModel } from '../types/dougongAssembly';

const group001Assembly: DougongAssemblyModel = {
  modelId: '1',
  sourceAsset: '/assembly/group_001/model.obj',
  parts: [
    {
      id: 'base-block',
      name: '基础斗与承托木',
      nodeNames: ['qingmo_obj_619', 'qingmo_obj_618', 'qingmo_obj_617', 'qingmo_obj_616', 'qingmo_obj_615', 'qingmo_obj_613', 'qingmo_obj_612', 'qingmo_obj_608'],
      layer: 1,
      explodeOffset: { x: 0, y: -360, z: 0 },
      color: '#6f4a2d',
      description: '靠近底部的斗、托木与基础承重构件。',
    },
    {
      id: 'mid-brackets',
      name: '中层拱与横向构件',
      nodeNames: ['qingmo_obj_603', 'qingmo_obj_601', 'qingmo_obj_600', 'qingmo_obj_599', 'qingmo_obj_597', 'qingmo_obj_596', 'qingmo_obj_595', 'qingmo_obj_594'],
      layer: 2,
      explodeOffset: { x: -360, y: 0, z: 0 },
      color: '#9b4f35',
      description: '中层横向出跳构件，负责向外承托。',
    },
    {
      id: 'cantilever-arms',
      name: '出跳拱臂',
      nodeNames: ['qingmo_obj_593', 'qingmo_obj_592', 'qingmo_obj_591', 'qingmo_obj_589', 'qingmo_obj_587', 'qingmo_obj_586', 'qingmo_obj_585', 'qingmo_obj_584'],
      layer: 3,
      explodeOffset: { x: 360, y: 0, z: 0 },
      color: '#3f7661',
      description: '向外挑出的主要拱臂与撑托构件。',
    },
    {
      id: 'top-members',
      name: '上层压板与收头构件',
      nodeNames: ['qingmo_obj_583', 'qingmo_obj_582', 'qingmo_obj_581', 'qingmo_obj_580', 'qingmo_obj_579', 'qingmo_obj_577', 'qingmo_obj_576', 'qingmo_obj_569', 'qingmo_obj_568', 'qingmo_obj_567', 'qingmo_obj_565', 'qingmo_obj_564', 'qingmo_obj_561'],
      layer: 4,
      explodeOffset: { x: 0, y: 360, z: 0 },
      color: '#b0803f',
      description: '上层压板、收头与顶部完成构件。',
    },
    {
      id: 'ornaments',
      name: '补充与异名节点',
      nodeNames: ['???', '???_1', '???_2', '???_3', '???_4', '???_5', '???_6', '???_7', '???_8', '???_9', '???_10'],
      layer: 5,
      explodeOffset: { x: 0, y: 0, z: 360 },
      color: '#5e6f86',
      description: '导出时名称缺失的节点，首版集中作为补充部件处理。',
    },
  ],
  steps: [
    { id: 'step-1', label: '搭建基础层', partIds: ['base-block'], order: 1, durationMs: 900 },
    { id: 'step-2', label: '安装中层拱件', partIds: ['mid-brackets'], order: 2, durationMs: 900 },
    { id: 'step-3', label: '拼接出跳拱臂', partIds: ['cantilever-arms'], order: 3, durationMs: 900 },
    { id: 'step-4', label: '收合顶部构件', partIds: ['top-members'], order: 4, durationMs: 900 },
    { id: 'step-5', label: '补齐细节节点', partIds: ['ornaments'], order: 5, durationMs: 800 },
  ],
};

export const DOUGONG_ASSEMBLY_MODELS: Record<string, DougongAssemblyModel> = {
  [group001Assembly.modelId]: group001Assembly,
};

export function getDougongAssemblyModel(modelId: string) {
  return DOUGONG_ASSEMBLY_MODELS[modelId] ?? null;
}
