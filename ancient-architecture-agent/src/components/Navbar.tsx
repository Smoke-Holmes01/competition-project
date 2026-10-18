import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Settings } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  const navLinks = [
    { name: '首页', path: '/' },
    { name: '展厅', path: '/workbench' },
    { name: '模型库', path: '/library' },
    { name: '关于', path: '/about' },
  ];

  const workbenchTarget = '/workbench';

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 flex items-center px-8 transition-all duration-300 bg-surface/90 backdrop-blur-xl shadow-[0px_12px_32px_-4px_rgba(139,90,43,0.15)] border-b border-accent-weak/30">
      <div className="flex justify-between items-center w-full max-w-[1440px] mx-auto">
        <div className="flex items-center gap-12">
          <Link to="/" className="flex flex-col items-start">
            <span className="text-xl font-bold tracking-tighter font-headline text-accent-strong">智构千年</span>
            <span className="text-xs text-on-surface/60 font-medium">Agent赋能的古建筑斗拱数字化保护与智能交互平台</span>
          </Link>
          <nav className="hidden md:flex items-center space-x-8 font-headline text-sm font-semibold tracking-tight">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`transition-all ${
                  location.pathname === link.path
                    ? 'text-accent-strong border-b-2 border-accent-strong pb-1'
                    : 'text-on-surface/60 hover:text-accent-strong'
                }`}
              >
                {link.name}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <Link to={workbenchTarget} className="ancient-button text-sm font-semibold transition-all hover:opacity-90 active:scale-95 shadow-sm">
            进入展厅
          </Link>
          <div className="flex items-center gap-2">
            <button type="button" className="p-2 rounded-md transition-colors hover:bg-surface-container-low text-on-surface-variant">
              <Settings size={20} />
            </button>

          </div>
        </div>
      </div>
    </header>
  );
}
