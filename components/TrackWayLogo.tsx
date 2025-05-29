import React from 'react';

interface TrackWayLogoProps {
  className?: string;
  style?: React.CSSProperties;
}

export const TrackWayLogo: React.FC<TrackWayLogoProps> = ({ className, style }) => {
  return (
    <svg 
      viewBox="0 0 220 100" // Adjusted viewBox for new infinity shape
      xmlns="http://www.w3.org/2000/svg" 
      className={className}
      style={style}
      aria-label="TrackWay Logo"
    >
      <defs>
        <linearGradient id="logoGradient" x1="0%" y1="50%" x2="100%" y2="50%">
          <stop offset="0%" style={{ stopColor: 'var(--color-brand-teal-start, #2DD4BF)', stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: 'var(--color-brand-blue-start, #60A5FA)', stopOpacity: 1 }} />
        </linearGradient>
        
        <filter id="subtleDropShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="1" result="blur"/>
          <feOffset in="blur" dx="1" dy="1.5" result="offsetBlur"/>
          <feMerge>
            <feMergeNode in="offsetBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        <style>
          {`
            @keyframes pulseEye {
              0%, 100% { opacity: 0.7; transform: scale(1); }
              50% { opacity: 1; transform: scale(1.05); }
            }
            .eye-animated { 
              animation: pulseEye 2.5s infinite ease-in-out;
              transform-origin: 155px 50px; /* Center of the eye */
            }
          `}
        </style>
      </defs>
      
      {/* Main Infinity Path */}
      <path 
        d="M110,50 C60,0 20,0 20,50 C20,100 60,100 110,50 C160,100 200,100 200,50 C200,0 160,0 110,50 Z"
        fill="none" 
        stroke="url(#logoGradient)" 
        strokeWidth="12"
        strokeLinecap="round"
        strokeLinejoin="round"
        filter="url(#subtleDropShadow)"
      />
      
      {/* E part - Horizontal Bar */}
      {/* Positioned within the left loop (center x=65) */}
      <line 
        x1="48" y1="50" x2="82" y2="50" 
        stroke="url(#logoGradient)" 
        strokeWidth="10" 
        strokeLinecap="round" 
        filter="url(#subtleDropShadow)"
      />
      
      {/* E part - Sound Waves (3 arcs, more prominent) */}
      {/* Emanating from top-left of E-loop (approx. peak (65,0), leftmost (20,50)) */}
      <g fill="none" stroke="url(#logoGradient)" strokeLinecap="round" filter="url(#subtleDropShadow)">
        <path d="M42 24 Q 52 30, 42 36" strokeWidth="3.5" /> {/* Inner */}
        <path d="M36 19 Q 52 30, 36 41" strokeWidth="4.5" /> {/* Middle */}
        <path d="M30 14 Q 52 30, 30 46" strokeWidth="5.5" /> {/* Outer */}
      </g>

      {/* C part - Eye */}
      {/* Positioned within the right loop (center x=155) */}
      <g className="eye-animated" filter="url(#subtleDropShadow)">
        <ellipse 
          cx="155" cy="50" rx="20" ry="13" 
          fill="none" 
          stroke="url(#logoGradient)" 
          strokeWidth="5" 
        />
        <ellipse 
          cx="155" cy="50" rx="7" ry="5" 
          fill="url(#logoGradient)" 
        />
        {/* Lines representing visual impairment / sensor scan - using semi-transparent white */}
        <line x1="140" y1="46" x2="170" y2="46" stroke="rgba(255,255,255,0.65)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="140" y1="50" x2="170" y2="50" stroke="rgba(255,255,255,0.65)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="140" y1="54" x2="170" y2="54" stroke="rgba(255,255,255,0.65)" strokeWidth="2.5" strokeLinecap="round" />
      </g>
    </svg>
  );
};
