
import React from 'react';
import AnimatedElement from './AnimatedElement'; // Added import
// Note: The `Feature` type used here will be slightly different from the one in `types.ts`
// as it will expect already translated strings for title, description, and linkText.
// The parent component (FeaturesSection) will handle the translation.

interface FeatureCardGrokProps {
  feature: {
    id: string;
    icon?: React.ReactElement<{ className?: string }>;
    title: string; // Already translated
    description: string; // Already translated
    linkText?: string; // Already translated "Saiba Mais" or similar
    imagePlaceholderClass?: string;
    linkUrl?: string;
  };
  index: number;
}

export const FeatureCardGrok: React.FC<FeatureCardGrokProps> = ({ feature, index }) => {
  let accentColorRgb = 'var(--color-brand-accent-glow-rgb)';
  if (feature.imagePlaceholderClass?.includes('sky') || feature.imagePlaceholderClass?.includes('blue')) {
    accentColorRgb = '0, 191, 255';
  } else if (feature.imagePlaceholderClass?.includes('purple') || feature.imagePlaceholderClass?.includes('fuchsia')) {
    accentColorRgb = '192, 132, 252';
  } else if (feature.imagePlaceholderClass?.includes('teal') || feature.imagePlaceholderClass?.includes('green')) {
    accentColorRgb = '20, 211, 169';
  } else if (feature.imagePlaceholderClass?.includes('rose') || feature.imagePlaceholderClass?.includes('red')) {
    accentColorRgb = '251, 113, 133';
  } else if (feature.imagePlaceholderClass?.includes('amber') || feature.imagePlaceholderClass?.includes('yellow')) {
    accentColorRgb = '251, 191, 36';
  } else if (feature.imagePlaceholderClass?.includes('indigo') || feature.imagePlaceholderClass?.includes('violet')) {
    accentColorRgb = '129, 140, 248';
  }

  return (
    <AnimatedElement
      initialClasses="opacity-0 translate-y-10"
      finalClasses="opacity-100 translate-y-0"
      transitionClasses={`transition-all duration-700 ease-out`}
      style={{transitionDelay: `${index * 100 + 300}ms`}}
      className="h-full"
    >
      <div className="group bg-brand-surface/70 backdrop-blur-lg rounded-xl overflow-hidden shadow-xl hover:shadow-brand-accent-glow/25 transition-all duration-300 transform hover:-translate-y-1.5 h-full flex flex-col border border-brand-surface-alt/60 hover:border-[rgba(var(--accent-color-rgb,var(--color-brand-accent-glow-rgb)),0.5)]"
           style={{ '--accent-color-rgb': accentColorRgb } as React.CSSProperties}
      >
        <div className={`h-48 md:h-56 w-full ${feature.imagePlaceholderClass || 'bg-brand-surface-alt'} relative overflow-hidden`}>
           <div 
             className="absolute inset-0 opacity-60 transition-opacity duration-400 group-hover:opacity-80" 
             style={{
               backgroundImage: `radial-gradient(circle at 20% 20%, rgba(var(--accent-color-rgb,var(--color-brand-accent-glow-rgb)),0.35) 0%, transparent 60%), 
                               radial-gradient(circle at 80% 80%, rgba(var(--accent-color-rgb,var(--color-brand-accent-subtle-rgb)),0.25) 0%, transparent 50%)`,
               backgroundSize: '150% 150%',
               animation: 'auroraBorealis 20s infinite ease-in-out alternate',
             }}
           ></div>
           <div className="absolute inset-0 opacity-10 group-hover:opacity-15 transition-opacity duration-300" style={{backgroundImage: 'radial-gradient(circle, rgba(var(--color-brand-primary-text-rgb, 240,240,245),0.1) 0.5px, transparent 0.5px)', backgroundSize: '6px 6px'}}></div>
            {feature.icon && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-50 group-hover:opacity-100 transition-opacity duration-300 transform group-hover:scale-110">
                {React.cloneElement(feature.icon, {className: "w-12 h-12 text-white/70 group-hover:text-white"})}
              </div>
            )}
        </div>
        
        <div className="p-6 md:p-7 flex-grow flex flex-col justify-between">
          <div>
            <h3 className="text-lg lg:text-xl font-bold text-brand-primary-text mb-2.5 group-hover:text-[rgb(var(--accent-color-rgb,var(--color-brand-accent-highlight-rgb)))] transition-colors duration-200">
              {feature.title}
            </h3>
            <p className="text-sm text-brand-secondary-text leading-relaxed mb-5">
              {feature.description}
            </p>
          </div>
          {feature.linkUrl && feature.linkText && (
            <a 
              href={feature.linkUrl} 
              className="inline-flex items-center text-xs font-semibold text-[rgb(var(--accent-color-rgb,var(--color-brand-accent-subtle-rgb)))] hover:text-[rgb(var(--accent-color-rgb,var(--color-brand-accent-highlight-rgb)))] transition-colors duration-200 mt-auto group/link"
              aria-label={`${feature.linkText} ${feature.title}`} // Dynamic aria-label
            >
              {feature.linkText}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5 ml-1.5 transform transition-transform group-hover/link:translate-x-0.5 duration-200">
                <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
              </svg>
            </a>
          )}
        </div>
      </div>
    </AnimatedElement>
  );
};
