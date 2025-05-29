
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconAutonomy, IconSafety, IconQualityOfLife, IconInnovation } from '../constants';
import { useLanguage } from '../i18n/LanguageContext';

interface BenefitPointProps {
  icon: React.ReactElement<{ className?: string }>; 
  titleKey: string;
  descriptionKey: string;
  delay: string;
}

const BenefitPoint: React.FC<BenefitPointProps> = ({ icon, titleKey, descriptionKey, delay }) => {
  const { t } = useLanguage();
  return (
    <AnimatedElement
      initialClasses="opacity-0 translate-x-[-25px]"
      finalClasses="opacity-100 translate-x-0"
      transitionClasses="transition-all duration-700 ease-out"
      style={{ transitionDelay: delay }}
      className="flex items-start space-x-5 group p-1"
    >
      <div className="flex-shrink-0 w-12 h-12 bg-brand-surface/80 backdrop-blur-md rounded-xl flex items-center justify-center text-brand-accent-glow shadow-lg border border-brand-surface-alt/60 group-hover:border-brand-accent-glow/50 group-hover:bg-brand-surface-alt transition-all duration-300 transform group-hover:scale-105">
        {React.cloneElement(icon, { className: "w-6 h-6" })}
      </div>
      <div>
        <h4 className="text-lg font-semibold text-brand-primary-text mb-1.5 group-hover:text-brand-accent-highlight transition-colors duration-200">{t(titleKey)}</h4>
        <p className="text-brand-secondary-text text-[15px] leading-relaxed">{t(descriptionKey)}</p>
      </div>
    </AnimatedElement>
  );
};

export const WhatIsTrackieSection: React.FC = () => {
  const { t } = useLanguage();

  const benefitPointsMeta = [
    { icon: <IconAutonomy />, titleKey: "whatIsTrackieSection.benefitPoints.autonomy.title", descriptionKey: "whatIsTrackieSection.benefitPoints.autonomy.description", delay: "400ms" },
    { icon: <IconSafety />, titleKey: "whatIsTrackieSection.benefitPoints.safety.title", descriptionKey: "whatIsTrackieSection.benefitPoints.safety.description", delay: "550ms" },
    { icon: <IconQualityOfLife />, titleKey: "whatIsTrackieSection.benefitPoints.interaction.title", descriptionKey: "whatIsTrackieSection.benefitPoints.interaction.description", delay: "700ms" },
    { icon: <IconInnovation />, titleKey: "whatIsTrackieSection.benefitPoints.perception.title", descriptionKey: "whatIsTrackieSection.benefitPoints.perception.description", delay: "850ms" },
  ];

  return (
    <section id="what-is-trackie" className="py-28 md:py-40 bg-brand-bg content-over-noise overflow-hidden">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('whatIsTrackieSection.subtitle')}
          title={t('whatIsTrackieSection.title')}
          align="left"
          description={t('whatIsTrackieSection.description')}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 md:gap-20 items-center mt-16 md:mt-20">
          <AnimatedElement
            initialClasses="opacity-0 lg:translate-x-[-30px]"
            finalClasses="opacity-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-200ms"
          >
            <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-brand-primary-text mb-6" style={{textShadow: '0 0 10px rgba(var(--color-brand-accent-glow-rgb), 0.3)'}}>
              {t('whatIsTrackieSection.secondaryTitle')}
            </h3>
            <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-5">
              {t('whatIsTrackieSection.secondaryDescription1')}
            </p>
            <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-10">
              {t('whatIsTrackieSection.secondaryDescription2')}
            </p>
            <div className="space-y-8">
              {benefitPointsMeta.map((point) => (
                <BenefitPoint 
                  key={point.titleKey}
                  icon={point.icon} 
                  titleKey={point.titleKey} 
                  descriptionKey={point.descriptionKey}
                  delay={point.delay}
                />
              ))}
            </div>
          </AnimatedElement>

          <AnimatedElement
            initialClasses="opacity-0 scale-90 lg:translate-x-[30px]"
            finalClasses="opacity-100 scale-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-400ms"
            className="group aspect-square md:aspect-[5/4] rounded-2xl overflow-hidden shadow-2xl shadow-brand-accent-subtle/10 border border-brand-surface-alt/70"
          >
            <img
              src="https://source.unsplash.com/1000x1200/?abstract-neural-network-visualization,futuristic-assistive-technology,glowing-data-streams,minimalist-tech-concept"
              alt={t('whatIsTrackieSection.secondaryTitle')} // Example of translating alt text
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              loading="lazy"
            />
             <div className="absolute inset-0 bg-gradient-to-br from-brand-accent-glow/5 via-transparent to-brand-accent-subtle/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          </AnimatedElement>
        </div>
      </div>
    </section>
  );
};
