
import React from 'react';
import AnimatedElement from './AnimatedElement';
import { useLanguage } from '../i18n/LanguageContext';

export const VisionSection: React.FC = () => {
  const { t } = useLanguage();
  return (
    <section id="vision" className="relative py-32 md:py-48 bg-brand-bg text-brand-primary-text overflow-hidden content-over-noise">
      <div 
        aria-hidden="true"
        className="animate-lightBeamPulse absolute top-1/2 left-0 right-0 h-[350px] md:h-[400px] -translate-y-1/2 pointer-events-none opacity-10 md:opacity-15"
      >
        <div 
          className="w-full h-full"
          style={{
            background: `linear-gradient(to right, transparent 5%, rgba(var(--color-brand-accent-glow-rgb), 0.7) 50%, transparent 95%)`,
            filter: 'blur(100px)',
          }}
        />
      </div>
      
      <div 
        className="absolute inset-0 z-0 opacity-[0.05] pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(var(--color-brand-accent-highlight-rgb),0.5) 0.5px, transparent 0.5px)',
          backgroundSize: '50px 50px',
          animation: 'float 10s infinite linear alternate'
        }}
      ></div>

      <div className="container mx-auto px-6 lg:px-8 text-center relative z-10">
        <AnimatedElement
          initialClasses="opacity-0 translate-y-16"
          finalClasses="opacity-100 translate-y-0"
          transitionClasses="transition-all duration-1200 ease-[cubic-bezier(0.16,1,0.3,1)] delay-200ms"
        >
          <h2 className="text-base md:text-lg font-semibold uppercase tracking-wider text-brand-accent-subtle mb-6">
            {t('visionSection.subtitle')}
          </h2>
          <p 
            className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-extrabold text-brand-primary-text mb-10 md:mb-12 tracking-tighter leading-tight"
            style={{textShadow: '0 2px 5px rgba(0,0,0,0.5), 0 0 25px rgba(var(--color-brand-accent-glow-rgb),0.2)'}}
          >
            {t('visionSection.title')}
          </p>
          <p className="max-w-3xl xl:max-w-4xl mx-auto text-md md:text-lg lg:text-xl text-brand-secondary-text leading-relaxed md:leading-loose mb-16 md:mb-20">
            {t('visionSection.description')}
          </p>
          <a
            href="https://plataforma.gpinovacao.senai.br/plataforma/ideia/274056" 
            target="_blank"
            rel="noopener noreferrer"
            className="shine-effect group inline-block px-10 py-4 text-base md:text-lg font-semibold text-brand-bg bg-brand-accent-glow rounded-xl
                       shadow-xl shadow-brand-accent-glow/40
                       hover:bg-brand-accent-highlight hover:shadow-brand-accent-glow/60
                       transform hover:scale-105 active:scale-95 transition-all duration-300 
                       focus:outline-none focus-visible:ring-4 focus-visible:ring-brand-accent-glow/50"
          >
            {t('visionSection.joinButton')}
          </a>
        </AnimatedElement>
      </div>
    </section>
  );
};
