export interface DougongModel {
  id: string;
  name: string;
  dynasty: '清代' | '宋代';
  groupCode: string;
  description: string;
  imageUrl: string;
  glbPath: string;
}

const sampleImages: Record<string, string> = {
  '1': '/samples/group_001.png',
  '3': '/samples/group_003.png',
  '4': '/samples/group_004.png',
  '8': '/samples/group_008.png',
};

function model(id: number, groupCode: string, dynasty: '清代' | '宋代', name: string, description?: string): DougongModel {
  return {
    id: String(id),
    name,
    dynasty,
    groupCode,
    description: description ?? `${name}，${dynasty}斗拱模型资源，支持 GLB 三维预览与 OBJ fallback。`,
    imageUrl: sampleImages[String(id)] ?? sampleImages['1'],
    glbPath: `/models_glb/${id}.glb`,
  };
}

export const MOCK_MODELS: DougongModel[] = [
  model(1, 'group_001', '清代', '清代官式建筑中高等级的七踩斗拱', '这是清代官式建筑中高等级的七踩斗拱，从中心向四周出 3 层拽架，加上正心层共形成 7 排横拱，多用于宫殿、坛庙的主殿。结构精密，体现了清代官式建筑高度的标准化与模块化。'),
  model(2, 'group_002', '清代', '清代三踩斗拱'),
  model(3, 'group_003', '清代', '清式柱头科斗拱'),
  model(4, 'group_004', '宋代', '宋式补间铺作', '宋式补间铺作位于两柱之间、阑额之上，强调结构理性和卷杀柔美，是宋代铺作层的重要组成部分。'),
  model(5, 'group_005', '清代', '清代五踩斗拱'),
  model(6, 'group_006', '清代', '清代连廊 / 殿身面阔方向的平身科斗拱组', '清代连廊平身科斗拱组沿面阔方向串联，适配长面阔建筑檐部荷载，并形成连续的檐下秩序。'),
  model(7, 'group_007', '宋代', '宋式柱头铺作'),
  model(8, 'group_008', '清代', '清式五踩单昂平身科斗拱', '清式五踩单昂平身科斗拱出跳适中，昂身斜置，体现清代官式建筑标准化与装饰化的典型特征。'),
  model(9, 'group_009', '清代', '清式角科斗拱'),
  model(10, 'group_010', '清代', '清式柱头科斗拱'),
  model(11, 'group_011', '清代', '清式一斗二升交麻叶斗拱'),
  model(12, 'group_012', '宋代', '宋式五铺作转角铺作斗拱'),
  model(13, 'group_013', '清代', '清式九踩转角铺作斗拱'),
  model(14, 'group_014', '宋代', '宋式四铺作斗拱'),
  model(15, 'group_015', '宋代', '宋式七铺作转角/平身科斗拱'),
  model(16, 'group_016', '清代', '清式五踩转角铺作斗拱'),
  model(17, 'group_017', '清代', '清式三踩平身科斗拱'),
  model(18, 'group_018', '宋代', '宋式六铺作转角/平身科斗拱'),
  model(19, 'group_019', '清代', '清式七踩平身科/转角铺作斗拱'),
  model(20, 'group_020', '清代', '清式九踩重翘重昂转角铺作斗拱'),
  model(21, 'group_021', '清代', '清代官式斗拱'),
  model(22, 'group_022', '清代', '清式五踩重翘/重昂柱头科斗拱'),
  model(23, 'group_023', '清代', '清代官式三踩斗拱'),
  model(24, 'group_024', '清代', '清式五踩角科斗拱'),
  model(25, 'group_025', '清代', '清代官式高等级多层斗拱'),
  model(26, 'group_026', '清代', '清代官式超高层重檐斗拱'),
  model(27, 'group_027', '清代', '清代官式平身科五踩单翘单昂斗拱'),
  model(28, 'group_028', '清代', '清式三踩单昂平身科斗拱'),
  model(29, 'group_029', '清代', '清代三踩斗拱'),
  model(30, 'group_030', '清代', '清式一斗三升斗拱'),
  model(31, 'group_031', '宋代', '宋式四铺作单昂斗拱'),
  model(32, 'group_032', '宋代', '宋式五铺作单抄单昂计心造斗拱'),
  model(33, 'group_033', '清代', '清式五踩角科斗拱'),
  model(34, 'group_034', '宋代', '宋式六铺作单抄双下昂斗拱'),
  model(35, 'group_035', '清代', '清式五踩重昂斗拱'),
  model(36, 'group_036', '清代', '清代官式平身科九踩斗拱'),
  model(37, 'group_037', '清代', '清式五踩重昂平身科斗拱'),
  model(38, 'group_038', '清代', '清代官式平身科斗拱'),
  model(39, 'group_039', '清代', '清代官式平身科三踩斗拱'),
  model(40, 'group_040', '宋代', '宋式七铺作双抄双下昂斗拱'),
  model(41, 'group_041', '清代', '清代官式平身科九踩斗拱'),
  model(42, 'group_042', '清代', '清式九踩斗拱的艺术化摆件 / 结构模型'),
  model(43, 'group_043', '清代', '清式平身科三踩斗拱'),
  model(44, 'group_044', '清代', '清式九踩斗拱'),
  model(45, 'group_045', '清代', '清代官式平身科七踩斗拱'),
  model(46, 'group_046', '清代', '清式七踩重昂斗拱'),
];
