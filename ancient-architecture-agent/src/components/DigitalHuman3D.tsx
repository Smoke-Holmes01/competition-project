import React, { useRef, useEffect, useState, useMemo } from 'react';

interface DigitalHuman3DProps {
  isSpeaking: boolean;
  text?: string;
  height?: string | number;
  width?: string | number;
}

export default function DigitalHuman3D({
  isSpeaking,
  text = '',
  height = '100%',
  width = '100%'
}: DigitalHuman3DProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);

  const containerHeight = useMemo(() => {
    if (typeof height === 'number') return `${height}px`;
    return height;
  }, [height]);

  const containerWidth = useMemo(() => {
    if (typeof width === 'number') return `${width}px`;
    return width;
  }, [width]);

  useEffect(() => {
    if (videoRef.current) {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (videoRef.current) {
      if (isSpeaking) {
        videoRef.current.play().catch(err => {
          console.error('视频播放失败:', err);
        });
      } else {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    }
  }, [isSpeaking]);

  return (
    <div
      style={{
        height: containerHeight,
        width: containerWidth,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '8px',
        background: '#f5f5f0'
      }}
    >
      {isLoading ? (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          color: '#8B5A2B',
          width: '80%',
          textAlign: 'center'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '3px solid #D4A574',
            borderTopColor: '#8B5A2B',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <div style={{
            fontSize: '14px',
            fontWeight: '500'
          }}>
            正在加载数字人视频...
          </div>
        </div>
      ) : (
        <video
          ref={videoRef}
          src="/avaters/digital-human-video.mp4"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
          muted
          loop
        />
      )}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}