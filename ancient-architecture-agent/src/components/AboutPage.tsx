import React from 'react';
import { Link } from 'react-router-dom';
import { Compass, BookOpen, Award, ChevronRight } from 'lucide-react';

export default function AboutPage() {
  return (
    <section className="py-32 bg-surface ancient-section">
      <div className="max-w-4xl mx-auto px-8">
        <div className="text-center mb-16">
          <h1 className="text-3xl font-bold text-ink mb-4">关于智构千年</h1>
          <div className="w-20 h-1 bg-accent-strong mx-auto"></div>
        </div>

        <div className="space-y-12">
          {/* 项目背景 */}
          <div className="wooden-border p-8 paper-texture">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-accent-strong/10 rounded-full flex items-center justify-center">
                <Compass size={24} className="text-accent-strong" />
              </div>
              <h2 className="text-xl font-bold text-ink">项目背景</h2>
            </div>
            <p className="text-sm leading-relaxed text-on-surface/70">
              古建筑是中华优秀传统文化的重要载体，斗拱作为最具代表性的结构构件，承载着历代建筑技艺与审美智慧，是区分不同历史时期建筑风格的关键标识。然而，随着自然风化、人为破坏及修缮不当等因素影响，大量古建筑斗拱出现残缺、变形甚至消失，传统识别与展示方式高度依赖专家经验，效率低、门槛高、难以标准化，无法满足当前大规模数字化保护与公众文化传播的迫切需求。
            </p>
          </div>

          {/* 系统介绍 */}
          <div className="wooden-border p-8 paper-texture">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-accent-strong/10 rounded-full flex items-center justify-center">
                <BookOpen size={24} className="text-accent-strong" />
              </div>
              <h2 className="text-xl font-bold text-ink">系统介绍</h2>
            </div>
            <p className="text-sm leading-relaxed text-on-surface/70 mb-4">
              "智构千年" 是一款面向古建筑斗拱的数字化保护与智能交互平台。系统以智能体为核心驱动，用户只需上传古建筑图像，即可自动完成斗拱构件识别与历史朝代匹配。
            </p>
            <p className="text-sm leading-relaxed text-on-surface/70 mb-4">
              针对残缺、破损的斗拱图像，系统具备智能补全能力，能够基于局部特征推断完整结构形态。识别完成后，系统自动匹配对应朝代风格的三维斗拱模型，并通过增强现实技术以逐层动画形式动态展示从栌斗到散斗的完整搭建过程，支持多角度自由观察。
            </p>
            <p className="text-sm leading-relaxed text-on-surface/70">
              此外，系统还提供古建筑知识智能问答服务，用户可通过文本或图像输入获取专业的朝代特征、构造解析与历史背景解说。
            </p>
          </div>

          {/* 项目目标 */}
          <div className="wooden-border p-8 paper-texture">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-accent-strong/10 rounded-full flex items-center justify-center">
                <Award size={24} className="text-accent-strong" />
              </div>
              <h2 className="text-xl font-bold text-ink">项目目标</h2>
            </div>
            <p className="text-sm leading-relaxed text-on-surface/70">
              本系统致力于降低古建筑认知门槛，辅助文物保护与学术研究，为文化爱好者、建筑专业学生与研究人员、文博机构及教育工作者提供便捷、直观、沉浸式的数字化工具。通过人工智能与增强现实技术的融合创新，系统实现了从"图像输入"到"虚拟重建"再到"沉浸式体验"的全流程智能化，让千年斗拱的营造智慧在数字时代焕发新的生命力，助力中华优秀传统文化的传承与创新发展。
            </p>
          </div>

          {/* 行动按钮 */}
          <div className="text-center">
            <Link 
              to="/workbench" 
              className="inline-flex items-center gap-2 bg-accent-strong text-white px-6 py-3 rounded-lg font-bold transition-all hover:bg-accent-strong/90 shadow-md"
            >
              开始体验 <ChevronRight size={18} />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}