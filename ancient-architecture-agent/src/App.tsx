import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import RequireAuth from './auth/RequireAuth';

const LandingPage = lazy(() => import('./components/LandingPage'));
const ModelLibrary = lazy(() => import('./components/ModelLibrary'));
const Workbench = lazy(() => import('./components/Workbench'));
const LoginPage = lazy(() => import('./components/LoginPage'));
const AboutPage = lazy(() => import('./components/AboutPage'));

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

function AppContent() {
  const location = useLocation();
  const isFullscreenPage = location.pathname === '/workbench';

  return (
      <div className="min-h-screen flex flex-col">
        {!isFullscreenPage && <Navbar />}
        <main className="flex-1">
          <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-surface text-sm font-semibold text-ink">正在加载...</div>}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/library" element={<ModelLibrary />} />
              <Route path="/login" element={<LoginPage />} />
              <Route element={<RequireAuth />}>
                <Route path="/workbench" element={<Workbench />} />
              </Route>
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </Suspense>
        </main>
      </div>
  );
}
