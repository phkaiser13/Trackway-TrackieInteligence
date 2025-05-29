
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconObstacleAlert, IconGpsNavigation, IconInnovation, IconSafety } from '../constants'; 
import { useLanguage } from '../i18n/LanguageContext';

interface FeaturePillProps {
  text: string; // Expect translated string
  icon?: React.ReactElement<{ className?: string }>;
  delay: string;
}

const FeaturePill: React.FC<FeaturePillProps> = ({ text, icon, delay }) => (
  <AnimatedElement
    initialClasses="opacity-0 translate-y-4"
    finalClasses="opacity-100 translate-y-0"
    transitionClasses="transition-all duration-500 ease-out"
    style={{ transitionDelay: delay }}
    className="bg-brand-surface/80 backdrop-blur-sm text-brand-secondary-text px-4 py-3 rounded-lg text-[15px] font-medium shadow-lg hover:bg-brand-surface-alt hover:text-brand-primary-text transition-all duration-200 flex items-center gap-2.5 border border-brand-surface-alt/70 hover:border-brand-accent-subtle/50"
  >
    {icon && React.cloneElement(icon, {className: "w-5 h-5 text-brand-accent-subtle group-hover:text-brand-accent-highlight transition-colors"})}
    <span>{text}</span>
  </AnimatedElement>
);

export const HatConceptSection: React.FC = () => {
  const { t, tArray } = useLanguage();
  const featurePillTexts = tArray('hatConceptSection.featurePills');
  const featurePillsMeta = [
    { icon: <IconInnovation />, delay: "500ms" },
    { icon: <IconGpsNavigation />, delay: "600ms" },
    { icon: <IconObstacleAlert />, delay: "700ms" },
    { icon: <IconSafety />, delay: "800ms" },
  ];

  return (
    <section id="hat-concept" className="py-28 md:py-40 bg-brand-surface/30 content-over-noise overflow-hidden">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('hatConceptSection.subtitle')}
          title={t('hatConceptSection.title')}
          align="left"
          description={t('hatConceptSection.description')}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 md:gap-16 items-center mt-16 md:mt-20">
          <AnimatedElement
            initialClasses="opacity-0 scale-95 lg:translate-x-[-30px]"
            finalClasses="opacity-100 scale-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-200ms"
            className="group aspect-video md:aspect-[16/10] rounded-xl overflow-hidden shadow-2xl shadow-brand-accent-glow/10 border border-brand-surface-alt/70"
          >
            <img
              src="/DEMONSTS/HAT_DEMONST.png"
              alt={t('hatConceptSection.title')} // Translating alt text
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              loading="lazy"
            />
          </AnimatedElement>

          <AnimatedElement
            initialClasses="opacity-0 lg:translate-x-[30px]"
            finalClasses="opacity-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-400ms"
          >
            <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-brand-primary-text mb-6" style={{textShadow: '0 0 10px rgba(var(--color-brand-accent-subtle-rgb), 0.4)'}}>
              {t('hatConceptSection.secondaryTitle')}
            </h3>
            <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-5">
              {t('hatConceptSection.secondaryDescription1')}
            </p>
            <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-8">
              {t('hatConceptSection.secondaryDescription2')}
            </p>
            <div className="space-y-4 mb-10">
              {featurePillTexts.map((text, index) => (
                <FeaturePill 
                  key={index} 
                  text={text} 
                  icon={featurePillsMeta[index]?.icon} 
                  delay={featurePillsMeta[index]?.delay || `${500 + index * 100}ms`} 
                />
              ))}
            </div>
             <AnimatedElement
                initialClasses="opacity-0 translate-y-6"
                finalClasses="opacity-100 translate-y-0"
                transitionClasses="transition-all duration-700 ease-out delay-[1000ms]"
            >
                <blockquote className="p-5 bg-brand-accent-glow/5 rounded-lg border-l-4 border-brand-accent-glow">
                    <p 
                        className="text-md font-medium text-brand-accent-highlight leading-relaxed"
                        dangerouslySetInnerHTML={{ __html: t('hatConceptSection.antiSegregationQuote') }}
                    />
                </blockquote>
            </AnimatedElement>
          </AnimatedElement>
        </div>
      </div>
    </section>
  );
};
