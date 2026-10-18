import React, { useEffect, useRef, useState } from 'react';
import '@google/model-viewer';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Construction,
  SearchCode,
  Compass,
  Rotate3d,
  PlusCircle,
  Image as ImageIcon,
  X,
  Send,
  Bot,
  Settings,
  CloudCheck,
  UserCircle,
  RefreshCw,
  RotateCw,
  BarChart3,
  BrainCircuit,
  Layers3,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Cuboid,
  Camera,
} from 'lucide-react';
import { MOCK_MODELS } from '../constants';
import { type AgentChatResponse, type Conversation, type ProgressEvent, getConversation, deleteConversation, streamAgentChat } from '../api';
import type { ModelViewerElement } from '../model-viewer';
import { applyModelViewerWoodMaterial, bindModelViewerWoodMaterial } from '../modelMaterials';
import DougongAssemblyViewer from './DougongAssemblyViewer';
import ARAssemblyPreview from './ARAssemblyPreview';
import { getDougongAssemblyModel } from '../data/dougongAssembly';

interface ChatMessage {
  role: 'user' | 'agent';
  text: string;
  time: string;
  imageUrl?: string;
  imageName?: string;
  pending?: boolean;
}

type WorkbenchView = 'recognition' | 'model' | 'knowledge';
type ModelPreviewMode = 'default' | 'assembly';
const ACTIVE_CONVERSATION_KEY = 'ancient-agent-active-conversation-v2';
const EXAMPLE_QUESTIONS = ['这张图是什么？', '这个斗拱是哪个朝代的？', '清代斗拱与宋代的主要区别？', '解释“七踩”在斗拱中的含义。'];

function createConversationId() { return `conv_${crypto.randomUUID().replaceAll('-', '')}`; }
function activeConversationId() {
  try {
    window.localStorage.removeItem('ancient-agent-workbench-chat');
    window.localStorage.removeItem('ancient-agent-model-provider');
    return window.localStorage.getItem(ACTIVE_CONVERSATION_KEY) || createConversationId();
  } catch { return createConversationId(); }
}
function restoredMessages(conversation: Conversation): ChatMessage[] {
  return conversation.messages.map(message => ({
    role: message.role === 'assistant' ? 'agent' : 'user', text: message.content, time: message.time,
    imageUrl: message.assets[0] ? `/agent/assets/${message.assets[0].id}` : undefined,
    imageName: message.assets.length > 1 ? `${message.assets.length} 张图片` : message.assets[0]?.filename,
  }));
}

function normalizeModelId(modelId: string | null) {
  if (!modelId) {
    return '1';
  }
  return MOCK_MODELS.some((item) => item.id === modelId) ? modelId : '1';
}

export default function Workbench() {
  const [searchParams] = useSearchParams();
  const chatFileInputRef = useRef<HTMLInputElement>(null);
  const modelViewerRef = useRef<ModelViewerElement>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const [conversationId, setConversationId] = useState(activeConversationId);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRestoring, setIsRestoring] = useState(true);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imagePreview, setImagePreview] = useState(MOCK_MODELS[0].imageUrl);
  const [previewImage, setPreviewImage] = useState<{url: string; name?: string} | null>(null);
  const [agentResult, setAgentResult] = useState<AgentChatResponse | null>(null);
  const [activeView, setActiveView] = useState<WorkbenchView>('recognition');
  const [activeModelId, setActiveModelId] = useState(() => normalizeModelId(searchParams.get('model')));
  const [isAutoRotate, setIsAutoRotate] = useState(true);
  const [modelViewerKey, setModelViewerKey] = useState(0);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const [previewMode, setPreviewMode] = useState<ModelPreviewMode>('default');
  const [selectedAssemblyPartId, setSelectedAssemblyPartId] = useState<string | null>(null);
  const [assemblyStepIndex, setAssemblyStepIndex] = useState(-1);
  const [isAssemblyPlaying, setIsAssemblyPlaying] = useState(false);
  const [isAssemblyExploded, setIsAssemblyExploded] = useState(true);
  const [isArPreviewOpen, setIsArPreviewOpen] = useState(false);

  const activeModel = MOCK_MODELS.find((item) => item.id === activeModelId) ?? MOCK_MODELS[0];
  const assemblyConfig = getDougongAssemblyModel(activeModel.id);
  const supportsAssembly = activeView === 'model' && activeModel.id === '1' && !!assemblyConfig;
  const canOpenArPreview = activeView === 'model' && !!assemblyConfig;
  const previewUsesAssembly = supportsAssembly && previewMode === 'assembly' && !!assemblyConfig;
  const activeAssemblyStep = assemblyConfig && assemblyStepIndex >= 0 ? assemblyConfig.steps[assemblyStepIndex] ?? null : null;
  const activeAssemblyPart = assemblyConfig?.parts.find((part) => part.id === selectedAssemblyPartId) ?? null;
  const assemblyProgressLabel = activeAssemblyStep ? `${assemblyStepIndex + 1}/${assemblyConfig?.steps.length ?? 0}` : '未开始';
  const renderModelUrl = agentResult?.render_instruction?.model_url ?? activeModel.glbPath;
  const modelResourceLabel = previewUsesAssembly ? (assemblyConfig?.sourceAsset ?? renderModelUrl) : renderModelUrl;
  const previewPoster = imageFiles.length ? imagePreview : activeModel.imageUrl;
  const previewModeLabel = previewUsesAssembly ? '拆解 / 组装' : '普通预览';
  const previewBadgeClass = previewUsesAssembly ? 'bg-accent-strong text-white' : 'bg-white/80 text-on-surface/60';
  const previewDescription = previewUsesAssembly ? '基于第一组拆分 OBJ 的部件预览与逐层组装动画' : '普通 GLB 预览';
  const modelInfoText = `当前使用 GLB 模型资源：${renderModelUrl}`;

  const shouldBindModelViewerMaterial = activeView !== 'knowledge';

  const confidence = agentResult?.render_instruction?.confidence ?? agentResult?.tool_result?.confidence;
  const top5 = agentResult?.tool_result?.top5 ?? [];

  useEffect(() => {
    try { window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId); } catch { /* optional */ }
    const controller = new AbortController();
    let active = true;
    setIsRestoring(true);
    getConversation(conversationId, controller.signal).then(conversation => {
      if (!active) return;
      setMessages(conversation ? restoredMessages(conversation) : []);
      setAgentResult(conversation?.last_result ?? null);
      setProgress(conversation?.last_result?.status_trace ?? []);
      const restoredModel = conversation?.last_result?.render_instruction?.model_id ?? conversation?.last_result?.tool_result?.obj_id;
      if (restoredModel) setActiveModelId(String(restoredModel));
      const view = conversation?.last_result?.ui_instruction?.target_view;
      if (view) setActiveView(view);
    }).catch(caught => {
      if (active) setError(caught instanceof Error ? caught.message : '会话恢复失败');
    }).finally(() => { if (active) setIsRestoring(false); });
    return () => { active = false; controller.abort(); };
  }, [conversationId]);

  useEffect(() => () => requestAbortRef.current?.abort(), []);

  useEffect(() => {
    if (!shouldBindModelViewerMaterial) {
      return;
    }
    return bindModelViewerWoodMaterial(modelViewerRef.current);
  }, [modelViewerKey, renderModelUrl, shouldBindModelViewerMaterial]);

  useEffect(() => {
    const modelId = searchParams.get('model');
    if (!modelId) {
      return;
    }
    const nextModelId = normalizeModelId(modelId);
    if (nextModelId !== activeModelId) {
      setActiveModelId(nextModelId);
      setAgentResult(null);
      setPreviewMode('default');
      setSelectedAssemblyPartId(null);
      setAssemblyStepIndex(-1);
      setIsAssemblyPlaying(false);
      setIsAssemblyExploded(true);
    }
  }, [activeModelId, searchParams]);

  useEffect(() => {
    if (activeModel.id !== '1') {
      setPreviewMode('default');
      setSelectedAssemblyPartId(null);
      setAssemblyStepIndex(-1);
      setIsAssemblyPlaying(false);
      setIsAssemblyExploded(true);
      setIsArPreviewOpen(false);
    }
  }, [activeModel.id]);

  useEffect(() => {
    if (previewMode !== 'assembly') {
      setSelectedAssemblyPartId(null);
      setAssemblyStepIndex(-1);
      setIsAssemblyPlaying(false);
      setIsAssemblyExploded(true);
    }
  }, [previewMode]);

  useEffect(() => {
    if (!previewUsesAssembly) {
      return;
    }
    setIsAutoRotate(false);
  }, [previewUsesAssembly]);

  function currentTime() {
    return new Date().toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'});
  }

  async function handleClearConversation() {
    if (isSending || isRestoring) return;
    setIsSending(true);
    try {
      await deleteConversation(conversationId);
      handleNewConversation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '清空失败');
    } finally { setIsSending(false); }
  }

  function handleNewConversation() {
    setConversationId(createConversationId());
    setMessages([]);
    setChatInput('');
    setError('');
    setAgentResult(null);
    setProgress([]);
    setImageFiles([]);
    setPreviewImage(null);
    setImagePreview(MOCK_MODELS[0].imageUrl);
  }

  function handleFileChange(files?: File[] | FileList | null) {
    const nextFiles = Array.from(files ?? []);
    if (nextFiles.length > 5 || nextFiles.some(file => file.size > 10 * 1024 * 1024)) {
      setError('最多上传 5 张图片，每张不超过 10 MiB。');
      return;
    }
    setImageFiles(nextFiles);
    if (nextFiles[0]) {
      setImagePreview(URL.createObjectURL(nextFiles[0]));
    } else {
      setImagePreview(MOCK_MODELS[0].imageUrl);
    }
  }

  async function handleSend() {
    const content = chatInput.trim();
    if (!content || isSending || isRestoring || requestAbortRef.current) return;

    const uploadedImages = imageFiles;
    const uploadedImageUrl = uploadedImages[0] ? URL.createObjectURL(uploadedImages[0]) : undefined;
    const uploadedImageName = uploadedImages.length > 1
      ? `${uploadedImages.length} 张图片`
      : uploadedImages[0]?.name;

    setError('');
    setIsSending(true);
    setProgress([]);
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setChatInput('');
    setMessages((items) => [
      ...items,
      {
        role: 'user',
        text: content,
        time: currentTime(),
        imageUrl: uploadedImageUrl,
        imageName: uploadedImageName,
      },
      { role: 'agent', text: 'Agent 正在回答，请耐心等待。', time: currentTime(), pending: true },
    ]);
    setImageFiles([]);
    setImagePreview(MOCK_MODELS[0].imageUrl);

    try {
      let streamedAnswer = '';
      const result = await streamAgentChat(
        content,
        conversationId,
        uploadedImages.length ? uploadedImages : undefined,
        {
          onProgress: event => setProgress(items => [...items, event]),
          onAnswerDelta: (delta) => {
            streamedAnswer += delta;
            setMessages((items) => items.map((item) => {
              if (!item.pending) {
                return item;
              }
              return { ...item, text: streamedAnswer };
            }));
          },
        },
        controller.signal,
      );
      if (result.conversation_id && result.conversation_id !== conversationId) {
        setConversationId(result.conversation_id);
      }
      setAgentResult(result);
      setProgress(result.status_trace);
      const modelId = result.render_instruction?.model_id ?? result.tool_result?.obj_id;
      if (modelId) {
        setActiveModelId(String(modelId));
      }
      if (result.ui_instruction?.target_view) {
        setActiveView(result.ui_instruction.target_view);
      }
      setMessages((items) => items.map((item) => (
        item.pending ? { role: 'agent', text: result.answer, time: currentTime(), pending: false } : item
      )));
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Agent 请求失败';
      setError(message);
      setMessages((items) => items.map((item) => (
        item.pending ? { role: 'agent', text: `请求失败：${message}`, time: currentTime(), pending: false } : item
      )));
    } finally {
      requestAbortRef.current = null;
      setIsSending(false);
    }
  }

  function handleResetModelView() {
    if (previewUsesAssembly) {
      setSelectedAssemblyPartId(null);
      setAssemblyStepIndex(-1);
      setIsAssemblyPlaying(false);
      setIsAssemblyExploded(true);
      return;
    }
    setIsAutoRotate(true);
    setModelViewerKey((value) => value + 1);
  }

  function handleActivateAssemblyMode() {
    if (!supportsAssembly) {
      return;
    }
    setPreviewMode('assembly');
    setSelectedAssemblyPartId(null);
    setAssemblyStepIndex(-1);
    setIsAssemblyPlaying(false);
    setIsAssemblyExploded(true);
  }

  function handleActivateDefaultMode() {
    setPreviewMode('default');
  }

  function handleToggleAssemblyPlayback() {
    if (!assemblyConfig) {
      return;
    }
    if (isAssemblyPlaying) {
      setIsAssemblyPlaying(false);
      setIsAssemblyExploded(false);
      return;
    }
    setSelectedAssemblyPartId(null);
    setAssemblyStepIndex(-1);
    setIsAssemblyExploded(true);
    setIsAssemblyPlaying(true);
  }

  function handleFocusAssemblyPart(partId: string) {
    setIsAssemblyPlaying(false);
    setSelectedAssemblyPartId((current) => current === partId ? null : partId);
    setAssemblyStepIndex(-1);
    setIsAssemblyExploded(true);
  }

  function handleAssemblyPlaybackComplete() {
    setIsAssemblyPlaying(false);
    setIsAssemblyExploded(false);
  }

  function handleStepJump(nextIndex: number) {
    if (!assemblyConfig) {
      return;
    }
    const bounded = Math.max(-1, Math.min(nextIndex, assemblyConfig.steps.length - 1));
    setIsAssemblyPlaying(false);
    setSelectedAssemblyPartId(null);
    setAssemblyStepIndex(bounded);
    setIsAssemblyExploded(bounded < 0);
  }

  const handleToggleAutoRotate = () => {
    if (previewUsesAssembly) {
      return;
    }
    setIsAutoRotate((value) => !value);
  };

  function toolButtonClass(view: WorkbenchView) {
    const active = activeView === view;
    return `w-full flex items-center gap-3 px-3 py-2.5 text-xs font-medium rounded-lg transition-transform active:scale-95 ${
      active ? 'text-accent-strong bg-white shadow-sm' : 'text-on-surface/50 hover:bg-white/50'
    }`;
  }

  function tabButtonClass(view: WorkbenchView) {
    return `h-full text-xs px-1 transition-colors ${
      activeView === view
        ? 'border-b-2 border-accent-strong text-accent-strong font-bold'
        : 'text-on-surface/40 hover:text-ink font-medium'
    }`;
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-surface">
      <header className="flex justify-between items-center gap-4 px-4 h-16 w-full z-50 bg-surface/90 backdrop-blur-xl shadow-[0px_12px_32px_-4px_rgba(139,90,43,0.15)] border-b border-accent-weak/30 font-headline tracking-tight text-sm overflow-x-auto custom-scrollbar">
        <div className="flex shrink-0 items-center gap-6">
          <div className="flex flex-col items-start">
            <span className="text-accent-strong font-bold text-xl tracking-widest whitespace-nowrap">智构千年</span>
            <span className="text-xs text-on-surface/60 font-medium">Agent赋能的古建筑斗拱数字化保护与智能交互平台</span>
          </div>
          <div className="h-4 w-px bg-accent-weak/30"></div>
          <span className="text-on-surface font-medium whitespace-nowrap">展厅工作台</span>
        </div>
        <div className="flex shrink-0 items-center gap-6">
          <nav className="flex items-center gap-4 whitespace-nowrap">
            <Link to="/workbench" className="text-accent-strong border-b-2 border-accent-strong py-1">展厅</Link>
            <Link to="/library" className="text-on-surface/60 hover:text-accent-strong transition-colors">模型库</Link>
            <Link to="/" className="text-on-surface/60 hover:text-accent-strong transition-colors">返回首页</Link>
          </nav>
          <div className="hidden xl:flex items-center gap-4 ml-4">
            <div className="flex items-center gap-2 px-2 py-1 bg-surface-container-low rounded">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-strong"></span>
              <span className="text-[10px] uppercase tracking-wider text-on-surface/60">Agent 在线</span>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 bg-surface-container-low rounded">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-strong"></span>
              <span className="text-[10px] uppercase tracking-wider text-on-surface/60">视觉后端就绪</span>
            </div>
          </div>
          <div className="flex items-center gap-3 ml-2">
            <Settings size={18} className="text-on-surface/60 cursor-pointer hover:text-accent-strong" />
            <CloudCheck size={18} className="text-on-surface/60 cursor-pointer hover:text-accent-strong" />
            <UserCircle size={18} className="text-on-surface/60 cursor-pointer hover:text-accent-strong" />
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-x-auto overflow-y-hidden bg-surface custom-scrollbar">
        <div className="flex h-full min-w-[1180px]">
          <aside className="w-[260px] shrink-0 bg-surface-container-low border-r border-outline-variant/15 flex flex-col">
            <div className="p-4 flex items-center justify-between">
              <h2 className="text-sm font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                <Construction size={16} /> 工具
              </h2>
            </div>
            <nav className="px-2 space-y-1">
              <button onClick={() => setActiveView('recognition')} className={toolButtonClass('recognition')}>
                <SearchCode size={18} /> 图片识别（匹配模型）
              </button>
              <button onClick={() => setActiveView('knowledge')} className={toolButtonClass('knowledge')}>
                <Compass size={18} /> 知识检索（结构历史）
              </button>
              <button onClick={() => setActiveView('model')} className={toolButtonClass('model')}>
                <Rotate3d size={18} /> 3D 预览（GLB）
              </button>
            </nav>

            <div className="mt-8 px-4">
              <h3 className="text-[10px] uppercase tracking-widest text-on-surface/40 mb-3 font-bold">素材上传</h3>
              <label className="border-2 border-dashed border-outline-variant/30 rounded-lg p-6 flex flex-col items-center justify-center bg-surface-container/50 hover:bg-surface-container-high transition-colors cursor-pointer group wooden-border">
                <PlusCircle className="text-on-surface/20 group-hover:text-accent-strong mb-2" size={24} />
                <p className="text-[11px] text-on-surface/50 text-center">拖拽或选择斗拱图片</p>
                <input
                  className="hidden"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  onChange={(event) => {
                    handleFileChange(event.target.files);
                    event.currentTarget.value = '';
                  }}
                />
              </label>
              <div className="mt-3 flex items-center gap-2 p-2 bg-white rounded border border-outline-variant/10 shadow-sm wooden-border">
                <ImageIcon size={14} className="text-accent-strong" />
                <span className="text-[11px] text-ink truncate">
                  {imageFiles.length ? imageFiles.map((file) => file.name).join('，') : 'group_001.png'}
                </span>
                <X size={12} className="text-on-surface/40 ml-auto cursor-pointer" onClick={() => handleFileChange(undefined)} />
              </div>
            </div>
          </aside>

          <section className="min-w-[560px] flex-1 flex flex-col bg-surface overflow-hidden">
            <div className="h-12 px-6 flex items-center gap-8 border-b border-outline-variant/10 bg-white/50 backdrop-blur">
              <button onClick={() => setActiveView('recognition')} className={tabButtonClass('recognition')}>识别结果</button>
              <button onClick={() => setActiveView('model')} className={tabButtonClass('model')}>三维模型</button>
              <button onClick={() => setActiveView('knowledge')} className={tabButtonClass('knowledge')}>知识摘要</button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
              {activeView !== 'knowledge' && (
                <div className="relative w-full min-h-[280px] aspect-[16/9] bg-surface-container-low rounded-lg overflow-hidden border border-outline-variant/10">
                  {previewUsesAssembly && assemblyConfig ? (
                    <DougongAssemblyViewer
                      config={assemblyConfig}
                      selectedPartId={selectedAssemblyPartId}
                      focusedStepIndex={assemblyStepIndex}
                      isExploded={isAssemblyExploded}
                      isPlaying={isAssemblyPlaying}
                      onPlayingChange={setIsAssemblyPlaying}
                      onStepChange={setAssemblyStepIndex}
                      onPlaybackComplete={handleAssemblyPlaybackComplete}
                    />
                  ) : (
                    <model-viewer
                      key={`${activeModel.id}-${modelViewerKey}`}
                      ref={modelViewerRef}
                      className="w-full h-full bg-surface-container-low"
                      src={renderModelUrl}
                      poster={previewPoster}
                      alt={activeModel.name}
                      camera-controls
                      auto-rotate={isAutoRotate ? true : undefined}
                      onLoad={() => applyModelViewerWoodMaterial(modelViewerRef.current)}
                    />
                  )}

                  <div className="absolute top-4 left-4 p-4 glass-panel rounded-lg shadow-xl w-[min(360px,calc(100%-2rem))] max-w-[360px] wooden-border">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span className="px-1.5 py-0.5 bg-accent-strong/10 text-accent-strong text-[10px] font-bold rounded">模型 ID: {activeModel.id}</span>
                      <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${previewBadgeClass}`}>{previewModeLabel}</span>
                      <span className="text-[10px] text-on-surface/50 font-medium truncate">资源 {modelResourceLabel}</span>
                    </div>
                    <h3 className="text-sm font-bold text-ink mb-1">{activeModel.name}</h3>
                    <p className="text-[10px] text-on-surface/50">{previewDescription}</p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-[11px] text-on-surface/50">识别置信度</span>
                      <span className="text-xs font-bold text-accent-strong">{(confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-surface-container h-1 rounded-full mt-1.5">
                      <div className="bg-accent-strong h-full rounded-full" style={{ width: `${Math.min(confidence * 100, 100).toFixed(1)}%` }}></div>
                    </div>
                    {previewUsesAssembly && assemblyConfig && (
                      <div className="mt-3 rounded-md bg-white/70 px-3 py-2 text-[11px] text-on-surface/60">
                        当前步骤：{activeAssemblyStep?.label ?? '拆解待播放'} · 已定位部件：{activeAssemblyPart?.name ?? '全部'}
                      </div>
                    )}
                  </div>

                  <div className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/10 bg-ink/90 p-1.5 shadow-2xl backdrop-blur-md">
                    <button onClick={handleResetModelView} className="p-2 text-white/70 transition-colors hover:text-white" title={previewUsesAssembly ? '重置拆解状态' : '重置视角'}>
                      <RefreshCw size={20} />
                    </button>
                    <div className="h-4 w-px bg-white/10"></div>
                    {previewUsesAssembly ? (
                      <>
                        <button
                          onClick={() => handleStepJump(assemblyStepIndex - 1)}
                          className="p-2 text-white/70 transition-colors hover:text-white"
                          title="上一步"
                        >
                          <SkipBack size={20} />
                        </button>
                        <button
                          onClick={handleToggleAssemblyPlayback}
                          className={`p-2 transition-colors ${isAssemblyPlaying ? 'text-accent-strong' : 'text-white/70 hover:text-white'}`}
                          title={isAssemblyPlaying ? '暂停组装动画' : '播放组装动画'}
                        >
                          {isAssemblyPlaying ? <Pause size={20} /> : <Play size={20} />}
                        </button>
                        <button
                          onClick={() => handleStepJump(assemblyStepIndex + 1)}
                          className="p-2 text-white/70 transition-colors hover:text-white"
                          title="下一步"
                        >
                          <SkipForward size={20} />
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={handleToggleAutoRotate}
                        className={`p-2 transition-colors ${isAutoRotate ? 'text-accent-strong' : 'text-white/70 hover:text-white'}`}
                        title={isAutoRotate ? '关闭自动旋转' : '开启自动旋转'}
                      >
                        <RotateCw size={20} />
                      </button>
                    )}
                  </div>

                  <div className="absolute bottom-20 right-4 z-10 rounded-lg border border-accent-strong/15 bg-accent-strong/5 px-3 py-2 text-[11px] leading-relaxed text-accent-strong wooden-border">
                    {previewUsesAssembly ? '拆解模式下可点击右侧部件列表进行单独预览。' : modelInfoText}
                  </div>
                </div>
              )}

              {activeView === 'recognition' && (
                <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)] gap-6">
                  <div className="space-y-4 min-w-0">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                      <BrainCircuit size={18} className="text-accent-strong" /> 识别摘要
                    </h4>
                    <div className="bg-white p-6 rounded-lg border border-outline-variant/10 min-w-0 wooden-border paper-texture">
                      <p className="text-sm leading-relaxed text-on-surface/70">
                        {agentResult ? (
                          <>工具调用：{agentResult.tool_calls?.length ? agentResult.tool_calls.map((item) => item.tool).join('，') : (agentResult.tool_called ? '已调用视觉匹配工具' : '未调用工具')}，当前模型 <code className="bg-surface-container-low px-1.5 rounded">{activeModel.groupCode}</code>：<span className="text-ink font-bold">{agentResult.tool_result?.model_name ?? activeModel.name}</span>。</>
                        ) : (
                          <>系统会按问题意图决定是否调用视觉匹配工具；涉及朝代、模型编号或三维预览时，将返回模型 ID、置信度与 GLB URL。</>
                        )}
                      </p>
                    </div>
                    {agentResult?.artifacts?.some((artifact) => artifact.type === 'image') && (
                      <div className="space-y-3">
                        <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                          <ImageIcon size={18} className="text-accent-strong" /> 工具产物
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {agentResult.artifacts.filter((artifact) => artifact.type === 'image').map((artifact) => (
                            <article key={artifact.id} className="rounded-lg border border-outline-variant/10 bg-white p-3 wooden-border">
                              {artifact.preview_url && (
                                <img
                                  src={artifact.preview_url}
                                  alt={artifact.label}
                                  className="h-36 w-full rounded object-contain bg-surface-container-low"
                                />
                              )}
                              <div className="mt-3 flex items-center justify-between gap-3">
                                <div className="min-w-0">
                                  <p className="truncate text-xs font-bold text-ink">{artifact.label}</p>
                                  <p className="mt-1 truncate font-mono text-[10px] text-on-surface/45">{artifact.url ?? artifact.download_url}</p>
                                </div>
                                <a
                                  href={artifact.download_url ?? artifact.url ?? '#'}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="shrink-0 rounded bg-accent-strong px-3 py-1.5 text-[11px] font-bold text-white transition-colors hover:bg-accent-strong/90"
                                >
                                  打开
                                </a>
                              </div>
                            </article>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4 min-w-0">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                      <BarChart3 size={18} className="text-accent-strong" /> 匹配列表 (Top 5)
                    </h4>
                    <div className="space-y-2">
                      {top5.map((item, i) => (
                        <div key={item.group} className={`p-3 border rounded-lg flex items-center justify-between gap-3 transition-colors ${i === 0 ? 'bg-white border-accent-strong' : 'bg-white/50 border-transparent text-on-surface/40'} wooden-border`}>
                          <div className="flex min-w-0 items-center gap-3">
                            <span className="text-[10px] font-bold">0{i + 1}</span>
                            <span className={`truncate text-xs font-bold ${i === 0 ? 'text-ink' : ''}`}>{item.group}</span>
                          </div>
                          <span className={`shrink-0 text-xs font-bold ${i === 0 ? 'text-accent-strong' : ''}`}>{(item.score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeView === 'model' && (
                <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(320px,0.85fr)] gap-6">
                  <div className="bg-white p-6 rounded-lg border border-outline-variant/10 wooden-border paper-texture">
                    <div className="flex items-center justify-between gap-3 mb-4">
                      <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                        <Rotate3d size={18} className="text-accent-strong" /> 三维模型
                      </h4>
                      {supportsAssembly && (
                        <div className="flex items-center gap-2 rounded-full border border-outline-variant/15 bg-surface-container-low p-1">
                          <button
                            type="button"
                            onClick={handleActivateDefaultMode}
                            className={`rounded-full px-3 py-1.5 text-[11px] font-bold transition-colors ${previewMode === 'default' ? 'bg-white text-accent-strong shadow-sm' : 'text-on-surface/55 hover:text-ink'}`}
                          >
                            普通预览
                          </button>
                          <button
                            type="button"
                            onClick={handleActivateAssemblyMode}
                            className={`rounded-full px-3 py-1.5 text-[11px] font-bold transition-colors ${previewMode === 'assembly' ? 'bg-accent-strong text-white shadow-sm' : 'text-on-surface/55 hover:text-ink'}`}
                          >
                            拆解组合
                          </button>
                        </div>
                      )}
                      {canOpenArPreview && (
                        <button
                          type="button"
                          onClick={() => setIsArPreviewOpen(true)}
                          className="flex items-center gap-1.5 rounded-full bg-ink px-3 py-2 text-[11px] font-bold text-white transition-colors hover:bg-ink/90"
                        >
                          <Camera size={14} /> AR
                        </button>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">模型 ID</p>
                        <p className="font-semibold text-ink">{activeModel.id}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">Group 编号</p>
                        <p className="font-semibold text-ink">{activeModel.groupCode}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">朝代</p>
                        <p className="font-semibold text-ink">{agentResult?.tool_result?.dynasty ?? activeModel.dynasty}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">置信度</p>
                        <p className="font-semibold text-accent-strong">{typeof confidence === 'number' ? `${(confidence * 100).toFixed(1)}%` : '未提供'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">预览模式</p>
                        <p className="font-semibold text-ink">{previewModeLabel}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">模式说明</p>
                        <p className="font-semibold text-ink">{previewDescription}</p>
                      </div>
                      <div className="col-span-2">
                        <p className="text-[10px] text-on-surface/40 font-bold uppercase tracking-wider mb-1">模型资源</p>
                        <p className="truncate font-mono text-xs text-accent-strong">{modelResourceLabel}</p>
                      </div>
                    </div>
                    {!supportsAssembly && (
                      <div className="mt-5 rounded-lg border border-outline-variant/10 bg-surface-container-lowest px-4 py-3 text-xs leading-relaxed text-on-surface/60">
                        拆解与组合首版当前仅支持第一组模型；其他模型仍使用原有 GLB 预览流程。
                      </div>
                    )}
                    {previewUsesAssembly && assemblyConfig && (
                      <div className="mt-5 space-y-4">
                        <div className="rounded-lg border border-accent-strong/15 bg-accent-strong/5 p-4">
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <p className="text-[10px] font-bold uppercase tracking-widest text-accent-strong">组装动画</p>
                              <p className="mt-1 text-sm font-semibold text-ink">{activeAssemblyStep?.label ?? '从拆解态开始播放'}</p>
                            </div>
                            <span className="rounded-full bg-white px-3 py-1 text-[11px] font-bold text-accent-strong">步骤 {assemblyProgressLabel}</span>
                          </div>
                          <div className="mt-4 flex flex-wrap items-center gap-2">
                            <button type="button" onClick={handleToggleAssemblyPlayback} className="flex items-center gap-1.5 rounded bg-accent-strong px-3 py-2 text-[11px] font-bold text-white transition-colors hover:bg-accent-strong/90">
                              {isAssemblyPlaying ? <Pause size={14} /> : <Play size={14} />} {isAssemblyPlaying ? '暂停播放' : '播放动画'}
                            </button>
                            <button type="button" onClick={handleResetModelView} className="flex items-center gap-1.5 rounded border border-outline-variant/15 px-3 py-2 text-[11px] font-bold text-ink transition-colors hover:border-accent-strong hover:text-accent-strong">
                              <Layers3 size={14} /> 回到拆解态
                            </button>
                          </div>
                        </div>
                        <div>
                          <div className="mb-3 flex items-center gap-2">
                            <Cuboid size={16} className="text-accent-strong" />
                            <p className="text-xs font-bold uppercase tracking-widest text-ink">部件列表</p>
                          </div>
                          <div className="space-y-2">
                            {assemblyConfig.parts.map((part) => {
                              const active = selectedAssemblyPartId === part.id;
                              return (
                                <button
                                  key={part.id}
                                  type="button"
                                  onClick={() => handleFocusAssemblyPart(part.id)}
                                  className={`w-full rounded-lg border px-4 py-3 text-left transition-colors ${active ? 'border-accent-strong bg-accent-strong/5' : 'border-outline-variant/10 bg-surface-container-lowest hover:border-accent-strong/40'}`}
                                >
                                  <div className="flex items-center justify-between gap-3">
                                    <div className="min-w-0">
                                      <div className="flex items-center gap-2">
                                        {part.color && (
                                          <span
                                            className="h-3 w-3 shrink-0 rounded-full border border-white shadow-sm"
                                            style={{ backgroundColor: part.color }}
                                          />
                                        )}
                                        <p className={`text-sm font-semibold ${active ? 'text-accent-strong' : 'text-ink'}`}>{part.name}</p>
                                      </div>
                                      <p className="mt-1 text-[11px] leading-relaxed text-on-surface/55">{part.description}</p>
                                    </div>
                                    <span className="shrink-0 rounded-full bg-white px-2 py-1 text-[10px] font-bold text-on-surface/50">第 {part.layer} 层</span>
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="bg-white p-6 rounded-lg border border-outline-variant/10 wooden-border paper-texture">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2 mb-4">
                      {previewUsesAssembly ? <Layers3 size={18} className="text-accent-strong" /> : <Cuboid size={18} className="text-accent-strong" />} {previewUsesAssembly ? '拆解说明' : '模型资源'}
                    </h4>
                    {previewUsesAssembly ? (
                      <div className="space-y-4 text-sm leading-relaxed text-on-surface/70">
                        <p>该模式使用第一组模型的拆分 OBJ 素材，将节点按 5 个部件组进行聚合，并支持逐层组装演示。</p>
                        <div className="rounded-lg bg-surface-container-low p-4 text-xs text-on-surface/60">
                          <p>当前部件：{activeAssemblyPart?.name ?? '全部部件'}</p>
                          <p className="mt-2">当前步骤：{activeAssemblyStep?.label ?? '尚未开始播放'}</p>
                          <p className="mt-2">说明：导出模型中存在部分 `???` 命名节点，首版统一归入“补充与异名节点”组。</p>
                        </div>
                      </div>
                    ) : (
                      <p className="break-all text-sm leading-relaxed text-on-surface/70">{modelInfoText}</p>
                    )}
                  </div>
                </div>
              )}

              {activeView === 'knowledge' && (
                <div className="space-y-4">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                    <Compass size={18} className="text-accent-strong" /> 知识摘要
                  </h4>
                  {agentResult?.rag_hits.length ? (
                    <div className="space-y-3">
                      {agentResult.rag_hits.map((hit, index) => (
                        <article key={hit.chunk_id} className="bg-white p-5 rounded-lg border border-outline-variant/10 wooden-border paper-texture">
                          <div className="mb-2 flex items-center justify-between gap-3">
                            <p className="text-[10px] font-bold uppercase tracking-widest text-accent-strong">[{index + 1}] {hit.source}</p>
                            {typeof hit.page === 'number' && <p className="text-[10px] text-on-surface/40">第 {hit.page} 页</p>}
                          </div>
                          <p className="text-sm leading-relaxed text-on-surface/70">{hit.text}</p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-white p-6 rounded-lg border border-outline-variant/10 wooden-border paper-texture">
                      <p className="text-sm leading-relaxed text-on-surface/70">{activeModel.description}</p>
                      <p className="mt-4 text-xs text-on-surface/50">右侧继续提问后，Agent 会结合知识库检索结果补充更具体的结构与历史信息。</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>

          <aside className="w-[360px] shrink-0 bg-white border-l border-outline-variant/15 flex flex-col">
            <div className="h-20 overflow-hidden px-4 py-0 flex items-center justify-between border-b border-outline-variant/10 bg-surface-container-low/30">
              <h2 className="text-sm font-bold uppercase tracking-widest text-ink flex items-center gap-2">
                <Bot size={20} className="text-accent-strong" /> Agent
              </h2>
              <div className="flex h-full items-center gap-3">
                <button
                  onClick={handleNewConversation}
                  disabled={isSending || isRestoring}
                  className="flex items-center gap-1.5 rounded border border-outline-variant/20 px-2.5 py-1.5 text-[11px] font-bold text-ink transition-colors hover:border-accent-strong hover:text-accent-strong disabled:cursor-not-allowed disabled:opacity-50 wooden-border"
                >
                  <PlusCircle size={14} /> 新对话
                </button>
                <button type="button" onClick={() => void handleClearConversation()} disabled={isSending || isRestoring}
                  className="text-[11px] text-on-surface/60 disabled:opacity-40">清空</button>
              </div>
            </div>
            <div className="border-b border-outline-variant/10 bg-white px-4 py-3">
              <p className="text-xs font-semibold text-accent-strong">DeepSeek V4.1 Flash · API</p>
              {isRestoring && <p className="mt-2 text-[11px] text-on-surface/60">正在恢复当前会话…</p>}
              {progress.length > 0 && <details open={isSending} className="mt-2 text-[11px] text-on-surface/60" aria-live="polite">
                <summary className="cursor-pointer">{isSending ? '执行进度' : '本轮执行记录'} · {progress.at(-1)?.message}</summary>
                <ol className="mt-2 max-h-36 overflow-auto space-y-1">
                  {progress.map((event, index) => <li key={index} className={event.event === 'tool_error' ? 'text-red-700' : ''}>
                    {index + 1}. {event.message}
                  </li>)}
                </ol>
              </details>}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
              {messages.length === 0 && (
                <div className="space-y-4">
                  <div className="rounded-lg border border-outline-variant/15 bg-surface-container-lowest p-4 wooden-border paper-texture">
                    <p className="text-xs font-bold text-ink">示例问题</p>
                    <div className="mt-3 grid gap-2">
                      {EXAMPLE_QUESTIONS.map((q) => (
                        <button
                          key={q}
                          onClick={() => setChatInput(q)}
                          className="w-full rounded border border-outline-variant/10 bg-white px-3 py-2 text-left text-[11px] leading-relaxed text-on-surface/70 transition-colors hover:border-accent-strong hover:text-accent-strong wooden-border"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} space-y-2`}>
                  <div className={`p-3 rounded-lg text-xs max-w-[90%] leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-ink text-white rounded-tr-none' : 'bg-surface-container-high text-on-surface rounded-tl-none'}`}>
                    {msg.imageUrl && (
                      <button
                        type="button"
                        onClick={() => setPreviewImage({ url: msg.imageUrl!, name: msg.imageName })}
                        className="mb-3 block overflow-hidden rounded-md border border-white/15 transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-white/40"
                      >
                        <img
                          src={msg.imageUrl}
                          alt={msg.imageName ?? '用户上传图片'}
                          className="max-h-48 w-auto max-w-[220px] object-cover"
                        />
                      </button>
                    )}
                    {msg.text.split('\n').map((line, j) => (
                      <React.Fragment key={j}>
                        {line}
                        <br />
                      </React.Fragment>
                    ))}
                  </div>
                  <span className="text-[9px] text-on-surface/40 uppercase font-medium">{msg.time} · {msg.role === 'user' ? '用户' : '系统生成'}</span>
                </div>
              ))}
            </div>

            <div className="p-4 border-t border-outline-variant/10 bg-white">
              <div className="relative bg-surface-container-high rounded-lg overflow-hidden transition-all focus-within:ring-1 ring-accent-strong/30 wooden-border">
                <textarea
                  className="w-full bg-transparent border-none text-xs p-3 focus:ring-0 placeholder:text-on-surface/40 resize-none paper-texture"
                  placeholder="问问我关于这个斗拱的细节..."
                  rows={2}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      void handleSend();
                    }
                  }}
                ></textarea>
                {error && <p className="px-3 pb-2 text-[11px] font-bold text-red-700">{error}</p>}
                <div className="flex items-center justify-between p-2 border-t border-outline-variant/5">
                  <div className="flex items-center gap-1">
                    <input
                      ref={chatFileInputRef}
                      className="hidden"
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      multiple
                      onChange={(event) => {
                        handleFileChange(event.target.files);
                        event.currentTarget.value = '';
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => chatFileInputRef.current?.click()}
                      className="p-1.5 text-on-surface/40 hover:text-accent-strong hover:bg-white rounded transition-colors"
                      title="上传图片"
                    >
                      <PlusCircle size={18} />
                    </button>
                    {imageFiles.length > 0 && (
                      <span className="max-w-[120px] truncate text-[10px] text-on-surface/50">
                        {imageFiles.length > 1 ? `${imageFiles.length} 张图片` : imageFiles[0].name}
                      </span>
                    )}
                  </div>
                  <button onClick={handleSend} disabled={isSending || isRestoring || !chatInput.trim()} className="flex items-center gap-2 px-3 py-1.5 bg-accent-strong text-white rounded text-[11px] font-bold uppercase tracking-wider hover:bg-accent-strong/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-wood-dark/20">
                    {isSending ? '分析中' : '发送'} <Send size={14} />
                  </button>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>

      <footer className="fixed bottom-0 w-full h-7 bg-ink border-t border-white/5 flex justify-between items-center px-4 z-50">
        <div className="flex items-center gap-6">
          <span className="text-white/40 font-body text-[9px] uppercase tracking-widest">Agent 8001 | DeepSeek API | BGE-M3</span>
        </div>
        <div className="flex items-center gap-4">
          <a className="text-white/40 hover:text-jade font-body text-[9px] uppercase tracking-widest" href="#">隐私政策</a>
          <a className="text-white/40 hover:text-jade font-body text-[9px] uppercase tracking-widest" href="#">文档</a>
          <a className="text-white/40 hover:text-jade font-body text-[9px] uppercase tracking-widest" href="#">反馈</a>
        </div>
      </footer>

      {previewImage && (
        <div className="fixed inset-0 z-[85] flex items-center justify-center bg-ink/80 px-4 backdrop-blur-sm" onClick={() => setPreviewImage(null)}>
          <div className="relative max-h-[92vh] max-w-[92vw]" onClick={(event) => event.stopPropagation()}>
            <button
              type="button"
              onClick={() => setPreviewImage(null)}
              className="absolute right-2 top-2 rounded-full bg-ink/75 p-2 text-white transition-colors hover:bg-ink"
              aria-label="关闭图片预览"
            >
              <X size={18} />
            </button>
            <img
              src={previewImage.url}
              alt={previewImage.name ?? '聊天图片预览'}
              className="max-h-[92vh] max-w-[92vw] rounded-lg bg-white object-contain shadow-2xl"
            />
          </div>
        </div>
      )}

      {isArPreviewOpen && assemblyConfig && (
        <ARAssemblyPreview
          config={assemblyConfig}
          modelName={activeModel.name}
          onClose={() => setIsArPreviewOpen(false)}
        />
      )}

    </div>
  );
}
