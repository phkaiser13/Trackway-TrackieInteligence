
import React from 'react';
import AnimatedElement from './AnimatedElement';
import { useLanguage } from '../i18n/LanguageContext';

const ScrollDownIndicator: React.FC = () => {
  const { t } = useLanguage();
  return (
    <AnimatedElement
      initialClasses="opacity-0 translate-y-3"
      finalClasses="opacity-70 translate-y-0 hover:opacity-100"
      transitionClasses="transition-all duration-1000 ease-out delay-[2000ms]"
      className="absolute bottom-8 md:bottom-12 left-1/2 -translate-x-1/2 cursor-pointer group"
    >
      <a href="#what-is-trackie" aria-label={t('hero.scrollDown')}>
        <svg className="w-7 h-7 md:w-8 md:h-8 text-brand-accent-subtle animate-float group-hover:text-brand-accent-highlight transition-colors duration-300" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M19 9l-7 7-7-7"></path>
        </svg>
      </a>
    </AnimatedElement>
  );
};

export const HeroSection: React.FC = () => {
  const { t } = useLanguage();

  return (
    <section id="hero" className="relative min-h-screen flex flex-col items-center justify-center bg-brand-bg text-brand-primary-text py-24 md:py-32 overflow-hidden">
      <div
        aria-hidden="true"
        className="aurora-bg animate-auroraBorealis absolute inset-0 pointer-events-none opacity-80"
        style={{ '--tw-gradient-from': 'rgba(var(--color-brand-accent-glow-rgb), 0.05)', '--tw-gradient-to': 'rgba(var(--color-brand-accent-highlight-rgb),0.03)' } as React.CSSProperties}
      />
      
      <div 
        aria-hidden="true" 
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{backgroundImage: 'linear-gradient(rgba(var(--color-brand-primary-text-rgb), 0.5) 1px, transparent 1px), linear-gradient(to right, rgba(var(--color-brand-primary-text-rgb), 0.5) 1px, transparent 1px)', backgroundSize: '40px 40px'}}
      ></div>

      <div className="container mx-auto px-6 lg:px-8 text-center relative z-10">
        <AnimatedElement 
            initialClasses="opacity-0"
            finalClasses="opacity-100" 
            transitionClasses=""
        >
          <h1 
            className="text-6xl sm:text-7xl md:text-8xl lg:text-[150px] xl:text-[170px] font-extrabold tracking-tighter leading-none animate-hero-title-reveal text-brand-primary-text"
            style={{
              animationDelay: '200ms',
              textShadow: `
                0px 1px 1px rgba(var(--color-brand-accent-glow-rgb),0.05),
                0px 0px 10px rgba(var(--color-brand-accent-glow-rgb),0.1),
                0 0 60px rgba(var(--color-brand-accent-highlight-rgb),0.1)
              `,
            }}
          >
            {t('hero.title')}
          </h1>
        </AnimatedElement>
        
        <AnimatedElement 
            initialClasses="opacity-0 translate-y-6" 
            finalClasses="opacity-100 translate-y-0" 
            transitionClasses="transition-all duration-1000 ease-out delay-[1000ms]"
        >
          <p className="mt-6 md:mt-8 text-lg md:text-xl lg:text-2xl font-medium text-brand-secondary-text max-w-3xl lg:max-w-4xl mx-auto animate-text-focus-in" style={{animationDelay: '1000ms'}}>
            {t('hero.tagline')}
          </p>
        </AnimatedElement>

        <AnimatedElement
          initialClasses="opacity-0 translate-y-6"
          finalClasses="opacity-100 translate-y-0"
          transitionClasses="transition-all duration-1000 ease-out delay-[1400ms]"
        >
          <div className="mt-10 md:mt-14">
            <a
              href="#what-is-trackie"
              className="shine-effect inline-block px-8 py-3.5 text-base md:text-lg font-semibold text-brand-bg bg-brand-accent-glow rounded-xl
                         shadow-lg shadow-brand-accent-glow/40
                         hover:bg-brand-accent-highlight hover:shadow-brand-accent-glow/60
                         transform hover:scale-105 active:scale-95 transition-all duration-300 
                         focus:outline-none focus-visible:ring-4 focus-visible:ring-brand-accent-glow/50 group"
            >
              {t('hero.exploreButton')}
            </a>
          </div>
        </AnimatedElement>
      </div>
      <ScrollDownIndicator />
    </section>
  );
};
