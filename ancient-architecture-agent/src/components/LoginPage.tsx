import React, { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { LockKeyhole, LogIn } from 'lucide-react';
import { useAuth } from '../auth/AuthProvider';

function useRedirectTarget() {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const redirect = params.get('redirect') || '/workbench';
  return redirect.startsWith('/') ? redirect : '/workbench';
}

export default function LoginPage() {
  const navigate = useNavigate();
  const { isAuthenticated, login } = useAuth();
  const redirect = useRedirectTarget();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  if (isAuthenticated) {
    return <Navigate to={redirect} replace />;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    setIsSubmitting(true);
    try {
      await login({ username: username.trim(), password });
      navigate(redirect, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : '登录失败，请稍后再试');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface px-6 py-24">
      <div className="mx-auto flex max-w-5xl overflow-hidden rounded-[28px] border border-accent-weak/30 bg-surface-container shadow-[0px_16px_42px_-12px_rgba(139,90,43,0.2)]">
        <section className="hidden w-1/2 flex-col justify-between bg-accent-strong px-10 py-12 text-white lg:flex">
          <div>
            <p className="text-sm font-semibold tracking-[0.2em] text-white/70">AUTHENTICATED ACCESS</p>
            <h1 className="mt-6 text-4xl font-bold font-headline leading-tight">启智构千年，共赏斗拱风华。</h1>
            <p className="mt-6 text-base leading-7 text-white/85">
              欢迎登录 “智构千年” 平台，开启中华古建筑的数字化探索之旅。
            </p>
          </div>
          <div className="rounded-2xl border border-white/15 bg-white/10 p-6 backdrop-blur-sm">
            <div className="flex items-center gap-3 text-sm font-medium text-white/90">
              <LockKeyhole size={18} />
              当前方案使用后端 HttpOnly Cookie 会话鉴权
            </div>
            <p className="mt-3 text-sm leading-6 text-white/75">
              登录态由服务端验证，前端只负责展示和跳转，不把敏感凭据存进 localStorage。
            </p>
          </div>
        </section>

        <section className="flex w-full flex-col justify-center px-8 py-12 lg:w-1/2 lg:px-12">
          <div className="mx-auto w-full max-w-md">
            <Link to="/" className="text-sm font-semibold text-accent-strong hover:opacity-80">返回首页</Link>
            <h2 className="mt-4 text-3xl font-bold font-headline text-on-surface">登录平台</h2>
            <p className="mt-3 text-sm leading-6 text-on-surface/70">
              输入账号和密码后即可进入受保护页面。若你是从其他页面跳转过来的，登录成功后会自动返回原目标页。
            </p>

            <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-on-surface">用户名</span>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className="w-full rounded-xl border border-accent-weak/40 bg-surface-container-lowest px-4 py-3 outline-none transition focus:border-accent-strong"
                  placeholder="请输入用户名"
                  autoComplete="username"
                  required
                />
              </label>

              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-on-surface">密码</span>
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full rounded-xl border border-accent-weak/40 bg-surface-container-lowest px-4 py-3 outline-none transition focus:border-accent-strong"
                  placeholder="请输入密码"
                  autoComplete="current-password"
                  required
                />
              </label>

              {error ? (
                <div className="rounded-xl border border-red-300/60 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              ) : null}

              <button
                type="submit"
                disabled={isSubmitting}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-accent-strong px-4 py-3 font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <LogIn size={18} />
                {isSubmitting ? '登录中...' : '登录并进入'}
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}
