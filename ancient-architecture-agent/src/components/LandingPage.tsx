import React from 'react';
import { ArrowRight, SearchCode, BrainCircuit, Box, ExternalLink, Upload, MessageSquare, Brain } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { MOCK_MODELS } from '../constants';
import { useAuth } from '../auth/AuthProvider';

export default function LandingPage() {
  const { isAuthenticated } = useAuth();
  const workbenchTarget = isAuthenticated ? '/workbench' : '/login?redirect=%2Fworkbench';

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative h-[870px] flex items-center overflow-hidden">
        <div className="absolute inset-0 z-0">
          <img
            className="w-full h-full object-cover"
            src="/images/home-hero.jpg"
            alt="Ancient Dougong"
            referrerPolicy="no-referrer"
          />
          <div className="absolute inset-0 bg-ink/40"></div>
        </div>
        <div className="relative z-10 max-w-[1240px] mx-auto px-8 w-full">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-2xl"
          >
            <h1 className="text-[3.5rem] font-bold leading-tight text-white mb-6 font-headline">
              让斗拱开口讲述自己的年代
            </h1>
            <p className="text-lg text-white/90 mb-10 font-body leading-relaxed">
              上传斗拱图像，智能体将结合视觉匹配、知识库检索与三维模型，为你还原古建筑构件的形制、朝代与结构脉络。
            </p>
            <div className="flex gap-4">
              <Link
                to={workbenchTarget}
                className="ancient-button px-8 py-4 rounded-lg font-semibold hover:brightness-110 transition-all flex items-center gap-2"
              >
                进入展厅 <ArrowRight size={18} />
              </Link>
              <Link
                to="/library"
                className="bg-white/10 backdrop-blur-md text-white border border-white/20 px-8 py-4 rounded-lg font-semibold hover:bg-white/20 transition-all"
              >
                浏览模型库
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section className="py-24 bg-surface ancient-section">
        <div className="max-w-[1240px] mx-auto px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {[
              {
                icon: <SearchCode className="text-accent-strong" size={24} />,
                title: '图像识别',
                desc: '通过斗拱图像分类与模型匹配，提取形制特征并返回模型编号',
              },
              {
                icon: <BrainCircuit className="text-accent-strong" size={24} />,
                title: '知识问答',
                desc: '结合古建斗拱知识库，解析构件历史背景、结构作用与形制特征',
              },
              {
                icon: <Box className="text-accent-strong" size={24} />,
                title: '3D 模型预览',
                desc: '支持 GLB 三维模型预览，方便旋转查看斗拱结构',
              },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
                className="flex flex-col gap-6 ancient-card"
              >
                <div className="w-16 h-16 flex items-center justify-center bg-accent-strong/10 rounded-lg border border-accent-weak/30">
                  {item.icon}
                </div>
                <h3 className="text-xl font-bold font-headline">{item.title}</h3>
                <p className="text-on-surface/70 leading-relaxed paper-texture">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="py-24 bg-surface-container-low ancient-section">
        <div className="max-w-[1240px] mx-auto px-8 text-center">
          <h2 className="text-3xl font-bold mb-4 font-headline">智慧分析流程</h2>
          <p className="text-on-surface/70 mb-16 max-w-2xl mx-auto paper-texture">
            简单图像问题直接给出基础解释；涉及朝代、模型编号、3D 渲染等深度问题时，智能体再调用视觉匹配工具
          </p>
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {[
              { icon: <Upload size={24} />, title: '上传图片' },
              { icon: <MessageSquare size={24} />, title: '提出问题' },
              { icon: <Brain size={24} />, title: '判断工具调用', ai: true },
              { icon: <Box size={24} />, title: '返回回答与 3D' },
            ].map((step, i) => (
              <React.Fragment key={i}>
                <div className="flex-1 flex flex-col items-center gap-4 w-full">
                  <div className="ancient-card p-6 w-full relative wooden-border">
                    {step.ai && (
                      <div className="absolute -top-3 -right-3">
                        <span className="flex h-6 w-6 rounded-full bg-accent-strong text-white items-center justify-center text-[10px] font-bold">AI</span>
                      </div>
                    )}
                    <div className={`${step.ai ? 'text-accent-strong' : 'text-accent-strong'} mb-2`}>{step.icon}</div>
                    <p className="font-semibold">{step.title}</p>
                  </div>
                </div>
                {i < 3 && <ArrowRight className="text-accent-weak hidden md:block" size={20} />}
              </React.Fragment>
            ))}
          </div>
        </div>
      </section>

      {/* Model Gallery */}
      <section className="py-24 bg-surface ancient-section">
        <div className="max-w-[1240px] mx-auto px-8">
          <div className="flex justify-between items-end mb-12">
            <div>
              <h2 className="text-3xl font-bold font-headline mb-2">斗拱模型</h2>
              <p className="text-on-surface/70">覆盖多种典型斗拱形制，以各朝代斗拱资源为核心</p>
            </div>
            <Link to="/library" className="text-accent-strong font-semibold flex items-center gap-2 hover:underline">
              查看全部模型 <ExternalLink size={16} />
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {MOCK_MODELS.slice(0, 4).map((model) => (
              <motion.div
                key={model.id}
                whileHover={{ y: -5 }}
                className="ancient-card overflow-hidden group museum-shadow wooden-border"
              >
                <div className="aspect-square relative overflow-hidden bg-surface-container">
                  <img
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                    src={model.imageUrl}
                    alt={model.name}
                    referrerPolicy="no-referrer"
                  />
                  <div className={`absolute top-3 left-3 ${model.dynasty === '清代' ? 'bg-accent-strong/80' : 'bg-accent-strong/80'} backdrop-blur-md text-white text-[10px] px-2 py-1 rounded-sm uppercase tracking-widest font-bold`}>
                    {model.dynasty}
                  </div>
                </div>
                <div className="p-6 paper-texture">
                  <h4 className="font-headline font-bold mb-1">{model.name}</h4>
                  <p className="text-xs text-on-surface/70 line-clamp-2">{model.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-32 bg-accent-strong text-white overflow-hidden relative ancient-section">
        <div className="absolute inset-0 opacity-10 grayscale">
          <img
            className="w-full h-full object-cover"
            src="/samples/group_004.png"
            alt="Blueprint"
            referrerPolicy="no-referrer"
          />
        </div>
        <div className="max-w-[1240px] mx-auto px-8 relative z-10 text-center">
          <h2 className="text-4xl md:text-5xl font-bold font-headline mb-8 tracking-tight">
            从一张斗拱图片开始，<br />进入古建构件的三维展厅。
          </h2>
          <Link
            to={workbenchTarget}
            className="inline-block bg-surface text-accent-strong px-12 py-5 rounded-lg font-bold text-lg hover:brightness-110 transition-all active:scale-95 shadow-xl border border-wood-dark/20"
          >
            进入展厅
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="w-full py-12 bg-surface-container-low text-on-surface ancient-section">
        <div className="max-w-[800px] mx-auto px-8 flex flex-col md:flex-row justify-between items-center gap-4 paper-texture">
          <div className="flex flex-col items-center">
            <span className="text-sm font-bold text-accent-strong">智构千年</span>
            <span className="text-xs text-on-surface/60 font-medium">Agent赋能的古建筑斗拱数字化保护与智能交互平台</span>
          </div>
          <p className="text-xs opacity-60">© 2026 智构千年. All rights reserved.</p>
          <div className="flex gap-6 text-xs opacity-60">
            <a className="hover:opacity-100 hover:text-accent-strong transition-all" href="#">隐私政策</a>
            <a className="hover:opacity-100 hover:text-accent-strong transition-all" href="#">服务条款</a>
            <a className="hover:opacity-100 hover:text-accent-strong transition-all" href="#">加入我们</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
