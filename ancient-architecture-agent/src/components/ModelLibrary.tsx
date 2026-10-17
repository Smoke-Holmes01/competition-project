import React, { useEffect, useRef, useState } from 'react';
import '@google/model-viewer';
import { Search, SortAsc, CheckCircle2, RefreshCw, RotateCw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { MOCK_MODELS, DougongModel } from '../constants';
import type { ModelViewerElement } from '../model-viewer';
import { applyModelViewerWoodMaterial, bindModelViewerWoodMaterial } from '../modelMaterials';

export default function ModelLibrary() {
  const modelViewerRef = useRef<ModelViewerElement>(null);
  const [selectedModel, setSelectedModel] = useState<DougongModel>(MOCK_MODELS[0]);
  const [filter, setFilter] = useState<'全部' | '清代' | '宋代' | '唐代' | '元代'| '明代'>('全部');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredModels = MOCK_MODELS.filter((model) => {
    const matchesFilter = filter === '全部' || model.dynasty === filter;
    const matchesSearch = model.name.includes(searchQuery) || model.groupCode.includes(searchQuery);
    return matchesFilter && matchesSearch;
  });

  useEffect(() => bindModelViewerWoodMaterial(modelViewerRef.current), [selectedModel.glbPath]);

  return (
    <main className="pt-24 pb-12 px-8 max-w-[1440px] mx-auto min-h-screen paper-texture">
      {/* Hero Header */}
      <section className="mb-12 flex flex-col md:flex-row md:items-end justify-between gap-8 ancient-section">
        <div className="max-w-2xl">
          <h1 className="text-4xl font-headline font-bold text-accent-strong mb-4">斗拱模型库</h1>
          <p className="text-on-surface/70 text-lg leading-relaxed mb-6 paper-texture">
            以各朝代斗拱资源为核心，浏览每一组模型的编号、朝代、形制与三维资源。
          </p>
          <div className="flex flex-wrap gap-3">
            <span className="px-3 py-1 bg-surface-container text-on-surface/70 text-xs font-semibold rounded-sm border border-accent-weak/30">GLB 模型</span>
            <span className="px-3 py-1 bg-accent-strong/10 text-accent-strong text-xs font-semibold rounded-sm border border-accent-strong/10">宋式 / 清式 / 唐式 / 元式 / 明式</span>
            <span className="px-3 py-1 bg-surface-container text-on-surface/70 text-xs font-semibold rounded-sm border border-accent-weak/30">支持 3D 预览</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-4">
          <div className="flex bg-surface-container-low p-1 rounded-md border border-accent-weak/30">
            {['全部', '清代', '宋代', '唐代', '元代', '明代'].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f as any)}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                  filter === f ? 'bg-accent-strong text-white shadow-sm' : 'text-on-surface/60 hover:text-on-surface'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
          <div className="text-on-surface/40 text-xs font-medium">当前 {filteredModels.length} 项</div>
        </div>
      </section>

      {/* Search & Utility Row */}
      <section className="mb-8 grid grid-cols-12 gap-6 items-center ancient-section">
        <div className="col-span-8 relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-on-surface/40" size={20} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-surface-container-high border border-accent-weak/30 rounded-md focus:ring-1 focus:ring-accent-strong/50 text-sm wooden-border"
            placeholder="搜索模型名称 / group 编号 / 朝代"
          />
        </div>
        <div className="col-span-4 flex justify-end">
          <button className="flex items-center gap-2 px-4 py-3 bg-surface-container-low text-on-surface/60 text-sm font-medium rounded-md hover:bg-surface-container-high transition-colors border border-accent-weak/30 wooden-border">
            <SortAsc size={18} />
            按模型编号
          </button>
        </div>
      </section>

      {/* Main Content Layout */}
      <section className="grid grid-cols-1 md:grid-cols-12 gap-8 h-[700px] ancient-section">
        {/* Left: Model List */}
        <div className="md:col-span-8 overflow-y-auto pr-4 space-y-3 custom-scrollbar">
          {filteredModels.map((model) => (
            <motion.div
              key={model.id}
              layout
              onClick={() => setSelectedModel(model)}
              className={`group flex items-center justify-between p-4 rounded-lg transition-all cursor-pointer ancient-border wooden-border ${
                selectedModel.id === model.id
                  ? 'bg-white border-l-4 border-accent-strong museum-shadow rounded-r-lg'
                  : 'bg-surface-container-low hover:bg-surface-container-high'
              }`}
            >
              <div className="flex items-center gap-6">
                <span className="text-xs font-mono font-bold text-on-surface/40">#{model.id.padStart(3, '0')}</span>
                <div className="paper-texture">
                  <h3 className={`font-headline font-semibold ${selectedModel.id === model.id ? 'text-accent-strong' : 'text-on-surface/80 group-hover:text-accent-strong'}`}>
                    {model.name}
                  </h3>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-on-surface/60">朝代: {model.dynasty}</span>
                    <span className="text-[10px] uppercase px-1.5 py-0.5 bg-surface-container text-on-surface/60 font-bold rounded-sm tracking-widest border border-accent-weak/30">
                      {model.groupCode}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className={`text-[10px] font-bold flex items-center gap-1 ${selectedModel.id === model.id ? 'text-accent-strong' : 'text-on-surface/40'}`}>
                  <CheckCircle2 size={14} fill={selectedModel.id === model.id ? 'currentColor' : 'none'} /> GLB Ready
                </span>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Right: Detail Pane */}
        <div className="md:col-span-4 ancient-card flex flex-col h-full overflow-hidden museum-shadow wooden-border">
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedModel.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col h-full"
            >
              {/* Preview Canvas */}
              <div className="relative h-1/2 bg-surface-container overflow-hidden group/canvas wooden-border">
                <model-viewer
                  ref={modelViewerRef}
                  className="w-full h-full cursor-grab bg-surface-container active:cursor-grabbing"
                  src={selectedModel.glbPath}
                  poster={selectedModel.imageUrl}
                  alt={selectedModel.name}
                  camera-controls
                  onLoad={(event) => applyModelViewerWoodMaterial(event.currentTarget as ModelViewerElement)}
                />
                <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
                {/* Canvas Tools */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
                  <button className="flex items-center gap-1.5 px-3 py-1.5 bg-surface/90 backdrop-blur-md text-on-surface text-[10px] font-bold rounded-full shadow-lg hover:bg-surface transition-all border border-accent-weak/30">
                    <RefreshCw size={14} /> 重置视角
                  </button>
                  <button className="flex items-center gap-1.5 px-3 py-1.5 bg-surface/90 backdrop-blur-md text-on-surface text-[10px] font-bold rounded-full shadow-lg hover:bg-surface transition-all border border-accent-weak/30">
                    <RotateCw size={14} /> 自动旋转
                  </button>
                </div>
                <div className="absolute top-4 right-4 px-2 py-1 bg-accent-strong text-white text-[10px] font-bold rounded-sm">3D 预览</div>
              </div>

              {/* Metadata & Info */}
              <div className="p-6 flex-1 overflow-y-auto custom-scrollbar paper-texture">
                <div className="mb-6">
                  <h2 className="text-2xl font-headline font-bold text-accent-strong mb-2">{selectedModel.name}</h2>
                  <div className="grid grid-cols-2 gap-y-3 gap-x-4">
                    <div>
                      <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-0.5">模型 ID</p>
                      <p className="text-sm font-semibold">{selectedModel.id}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-0.5">Group 编号</p>
                      <p className="text-sm font-semibold">{selectedModel.groupCode}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-0.5">朝代</p>
                      <p className="text-sm font-semibold">{selectedModel.dynasty}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-0.5">资源路径</p>
                      <p className="text-xs font-mono font-semibold text-accent-strong truncate">{selectedModel.glbPath}</p>
                    </div>
                  </div>
                </div>
                <div className="mb-8">
                  <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-2">详情摘要</p>
                  <p className="text-sm text-on-surface/70 leading-relaxed paper-texture">{selectedModel.description}</p>
                </div>
                <div className="flex flex-col gap-3">
                  <Link to={`/workbench?model=${selectedModel.id}`} className="w-full py-3 ancient-button font-headline font-bold text-sm transition-all hover:opacity-95 shadow-md text-center">
                    在展厅中使用
                  </Link>
                  <button
                    onClick={() => navigator.clipboard.writeText(selectedModel.glbPath)}
                    className="w-full py-3 bg-surface-container text-on-surface font-headline font-bold text-sm rounded-md transition-all hover:bg-surface-container-high flex items-center justify-center gap-2 border border-accent-weak/30 wooden-border"
                  >
                    复制模型路径
                  </button>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </section>

      {/* Footer Status Bar */}
      <footer className="fixed bottom-0 left-0 right-0 h-10 bg-surface-container-low border-t border-accent-weak/30 px-8 flex items-center justify-between z-50 ancient-section">
        <div className="flex items-center gap-6 text-[10px] text-on-surface/60 font-medium paper-texture">
          <span className="flex items-center gap-1.5">数据来源：知识库.pdf</span>
          <span className="flex items-center gap-1.5">资源格式：GLB / OBJ fallback</span>
        </div>
        <div className="text-[10px] font-mono text-on-surface/40">
          模型路径：/models_glb/{"{id}"}.glb
        </div>
      </footer>
    </main>
  );
}
