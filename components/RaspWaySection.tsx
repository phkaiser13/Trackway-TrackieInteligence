
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconInnovation } from '../constants'; 
import { useLanguage } from '../i18n/LanguageContext';

interface HighlightPointProps { 
  title: string; // Expect translated string
  children: React.ReactNode; // Content will be translated lines
  delay: string; 
  className?: string, 
  icon?: React.ReactNode 
}

const HighlightPoint: React.FC<HighlightPointProps> = ({ title, children, delay, className = '', icon }) => (
  <AnimatedElement
    initialClasses="opacity-0 transform-gpu translate-y-6"
    finalClasses="opacity-100 transform-gpu translate-y-0"
    transitionClasses="transition-all duration-700 ease-out"
    style={{ transitionDelay: delay }}
    className={`bg-brand-surface/60 backdrop-blur-lg p-6 rounded-xl shadow-xl border border-brand-surface-alt/50 hover:border-brand-accent-subtle/40 hover:shadow-brand-accent-subtle/15 transition-all duration-300 group ${className}`}
  >
    <div className="flex items-start">
      {icon && <div className="mr-4 mt-1 text-brand-accent-subtle group-hover:text-brand-accent-highlight transition-colors duration-200">{icon}</div>}
      <div>
        <h4 className="text-xl font-semibold text-brand-primary-text mb-2.5 group-hover:text-brand-accent-highlight transition-colors duration-200">{title}</h4>
        <div className="text-brand-secondary-text text-[15px] space-y-2 leading-relaxed">{children}</div>
      </div>
    </div>
  </AnimatedElement>
);

export const RaspWaySection: React.FC = () => {
  const { t, tArray } = useLanguage();

  const highlightPointsData = [
    { 
      icon: <IconInnovation className="w-6 h-6"/>, 
      titleKey: 'raspwaySection.highlightPoints.0.title', 
      descriptionLinesKeys: ['raspwaySection.highlightPoints.0.descriptionLines.0'], 
      delay: '500ms' 
    },
    { 
      titleKey: 'raspwaySection.highlightPoints.1.title', 
      descriptionLinesKeys: ['raspwaySection.highlightPoints.1.descriptionLines.0'], 
      delay: '650ms' 
    },
    { 
      titleKey: 'raspwaySection.highlightPoints.2.title', 
      descriptionLinesKeys: ['raspwaySection.highlightPoints.2.descriptionLines.0', 'raspwaySection.highlightPoints.2.descriptionLines.1'], 
      delay: '800ms',
      lastLineClass: "mt-2.5 text-xs text-brand-accent-subtle/90"
    },
  ];


  return (
    <section id="raspway-option" className="py-28 md:py-40 bg-brand-surface/30 content-over-noise overflow-hidden">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('raspwaySection.subtitle')}
          title={t('raspwaySection.title')}
          align="left"
          description={t('raspwaySection.description')}
        />

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 md:gap-16 items-center mt-16 md:mt-20">
          <AnimatedElement
            initialClasses="opacity-0 scale-90 lg:translate-x-[-30px]"
            finalClasses="opacity-100 scale-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-200ms"
            className="lg:col-span-2 group aspect-square md:aspect-[5/4] rounded-2xl overflow-hidden shadow-2xl shadow-brand-accent-glow/10 border border-brand-surface-alt/70"
          >
            <img
              src="/DEMONSTS/RASPWAY_DEMONST.png"
              alt={t('raspwaySection.title')} // Translating alt text
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              loading="lazy"
            />
            <div className="absolute inset-0 bg-gradient-to-br from-brand-accent-subtle/5 via-transparent to-brand-accent-glow/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          </AnimatedElement>

          <div className="lg:col-span-3 space-y-6 md:space-y-8">
             <AnimatedElement
                initialClasses="opacity-0"
                finalClasses="opacity-100"
                transitionClasses="transition-opacity duration-1000 ease-in delay-400ms"
            >
                <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-brand-primary-text mb-6" style={{textShadow: '0 0 10px rgba(var(--color-brand-accent-subtle-rgb), 0.4)'}}>
                  {t('raspwaySection.secondaryTitle')}
                </h3>
                <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-8">
                  {t('raspwaySection.secondaryDescription')}
                </p>
            </AnimatedElement>
            {highlightPointsData.map((point, index) => (
              <HighlightPoint 
                key={point.titleKey} 
                title={t(point.titleKey)} 
                delay={point.delay} 
                icon={point.icon}
              >
                {point.descriptionLinesKeys.map((lineKey, lineIndex) => (
                    <p key={lineKey} className={lineIndex === point.descriptionLinesKeys.length -1 && point.lastLineClass ? point.lastLineClass : ""}>
                        {t(lineKey)}
                    </p>
                ))}
              </HighlightPoint>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
