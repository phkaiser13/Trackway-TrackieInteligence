
import React from 'react';
import AnimatedElement from './AnimatedElement';

interface SectionTitleProps {
  subtitle?: string; // Expect translated string or undefined
  title: string;    // Expect translated string
  description?: string; // Expect translated string or undefined
  align?: 'left' | 'center' | 'right';
  titleAs?: 'h2' | 'h3';
  className?: string;
}

export const SectionTitle: React.FC<SectionTitleProps> = ({ 
  subtitle, 
  title, 
  description, 
  align = 'center',
  titleAs = 'h2',
  className = '',
}) => {
  const textAlignClass = {
    left: 'text-left items-start',
    center: 'text-center items-center',
    right: 'text-right items-end',
  }[align];

  const TitleComponent = titleAs;

  return (
    <AnimatedElement 
      className={`mb-16 md:mb-24 flex flex-col ${textAlignClass} content-over-noise ${className}`}
      initialClasses="opacity-0 translate-y-12"
      finalClasses="opacity-100 translate-y-0"
      transitionClasses="transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)] delay-100ms"
    >
      {subtitle && (
        <span className="text-sm md:text-base font-semibold uppercase tracking-wider text-brand-accent-subtle mb-3 md:mb-4">
          {subtitle}
        </span>
      )}
      <TitleComponent 
        className="text-4xl sm:text-5xl md:text-6xl lg:text-[64px] font-extrabold text-brand-primary-text mb-5 md:mb-7 tracking-tight leading-tight md:leading-tight"
        style={{textShadow: '0 1px 3px rgba(0,0,0,0.3), 0 0 15px rgba(var(--color-brand-accent-glow-rgb),0.1)'}}
      >
        {title}
      </TitleComponent>
      {description && (
        <p className={`max-w-3xl text-base md:text-lg text-brand-secondary-text leading-relaxed md:leading-loose ${align === 'center' ? 'mx-auto' : ''}`}>
          {description}
        </p>
      )}
    </AnimatedElement>
  );
};
