
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconMobileApp, IconInnovation } from '../constants'; 
import { useLanguage } from '../i18n/LanguageContext';

interface DetailCardProps { 
  title: string; // Expect translated string
  description: string; // Expect translated string
  delay: string, 
  icon?: React.ReactNode, 
  className?: string 
}

const DetailCard: React.FC<DetailCardProps> = ({ title, description, delay, icon, className = '' }) => (
    <AnimatedElement
        initialClasses="opacity-0 translate-y-8"
        finalClasses="opacity-100 translate-y-0"
        transitionClasses="transition-all duration-700 ease-out"
        style={{transitionDelay: delay}}
        className={`bg-brand-surface/60 backdrop-blur-lg p-6 rounded-xl shadow-xl border border-brand-surface-alt/50 hover:border-brand-accent-glow/40 hover:shadow-brand-accent-glow/15 transition-all duration-300 group ${className}`}
    >
        {icon && <div className="mb-4 text-brand-accent-glow group-hover:text-brand-accent-highlight transition-colors duration-200">{icon}</div>}
        <h4 className="text-xl font-semibold text-brand-primary-text mb-2.5 group-hover:text-brand-accent-highlight transition-colors duration-200">{title}</h4>
        <p className="text-brand-secondary-text text-[15px] leading-relaxed">{description}</p>
    </AnimatedElement>
);


export const SpotWaySection: React.FC = () => {
  const { t } = useLanguage();
  
  const detailCardsData = [
    { icon: <IconInnovation className="w-7 h-7"/>, titleKey: 'spotwaySection.detailCards.0.title', descriptionKey: 'spotwaySection.detailCards.0.description', delay: '500ms' },
    { icon: <IconMobileApp className="w-7 h-7"/>, titleKey: 'spotwaySection.detailCards.1.title', descriptionKey: 'spotwaySection.detailCards.1.description', delay: '650ms' },
    { titleKey: 'spotwaySection.detailCards.2.title', descriptionKey: 'spotwaySection.detailCards.2.description', delay: '800ms' },
  ];

  return (
    <section id="spotway-option" className="py-28 md:py-40 bg-brand-bg content-over-noise overflow-hidden">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('spotwaySection.subtitle')}
          title={t('spotwaySection.title')}
          align="right"
          description={t('spotwaySection.description')}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 md:gap-20 items-center mt-16 md:mt-20">
           <AnimatedElement
            initialClasses="opacity-0 lg:translate-x-[-30px]"
            finalClasses="opacity-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-400ms"
            className="order-2 lg:order-1"
          >
            <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-brand-primary-text mb-6" style={{textShadow: '0 0 10px rgba(var(--color-brand-accent-glow-rgb), 0.3)'}}>
              {t('spotwaySection.secondaryTitle')}
            </h3>
            <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed mb-8">
              {t('spotwaySection.secondaryDescription')}
            </p>
            <div className="space-y-6 md:space-y-8">
                 {detailCardsData.map(card => (
                     <DetailCard 
                        key={card.titleKey}
                        title={t(card.titleKey)}
                        description={t(card.descriptionKey)}
                        delay={card.delay}
                        icon={card.icon}
                    />
                 ))}
            </div>
          </AnimatedElement>

          <AnimatedElement
            initialClasses="opacity-0 scale-90 lg:translate-x-[30px]"
            finalClasses="opacity-100 scale-100 lg:translate-x-0"
            transitionClasses="transition-all duration-1000 ease-out delay-200ms"
            className="group aspect-square md:aspect-[5/4] rounded-2xl overflow-hidden shadow-2xl shadow-brand-accent-subtle/10 border border-brand-surface-alt/70 order-1 lg:order-2"
          >
            <img
              src="/DEMONSTS/SPOTWAY_DEMONST.png"
              alt={t('spotwaySection.title')} // Translating alt text
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
