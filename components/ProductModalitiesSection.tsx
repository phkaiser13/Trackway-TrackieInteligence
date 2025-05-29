
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconMobileApp } from '../constants'; 
import { LogoSpotWay, LogoRaspWay } from './AIEngineSection';
import { useLanguage } from '../i18n/LanguageContext';

interface ModalityCardProps {
  title: string; // Already translated string
  description: string; // Already translated string
  logoElement: React.ReactNode;
  linkHref: string;
  linkLabel: string; // Already translated string
  index: number; 
  tags?: string[]; // Array of already translated strings
  ariaLabelKeySuffix: string; // Suffix for aria-label translation key
}

const ModalityCard: React.FC<ModalityCardProps> = ({ title, description, logoElement, linkHref, linkLabel, index, tags, ariaLabelKeySuffix }) => {
  const { t } = useLanguage();
  return (
    <AnimatedElement
      initialClasses="opacity-0 translate-y-10"
      finalClasses="opacity-100 translate-y-0"
      transitionClasses="transition-all duration-700 ease-out"
      style={{ transitionDelay: `${index * 150 + 300}ms` }}
      className="group bg-brand-surface/70 backdrop-blur-lg p-6 md:p-8 rounded-xl shadow-xl h-full flex flex-col border border-brand-surface-alt/60 hover:border-brand-accent-glow/40 hover:shadow-brand-accent-glow/20 transition-all duration-300 transform hover:-translate-y-1"
    >
      <div className="flex justify-start mb-5 h-12 md:h-14 items-center">{logoElement}</div>
      <h4 className="text-xl md:text-2xl font-bold text-brand-primary-text mb-3 group-hover:text-brand-accent-highlight transition-colors duration-200 text-left">{title}</h4>
      <p className="text-brand-secondary-text text-sm md:text-[15px] leading-relaxed mb-6 flex-grow text-left">{description}</p>
      
      {tags && tags.length > 0 && (
        <div className="mb-5 flex flex-wrap gap-2">
          {tags.map(tag => (
            <span key={tag} className="text-xs bg-brand-surface-alt text-brand-accent-subtle px-2.5 py-1 rounded-full font-medium">
              {tag}
            </span>
          ))}
        </div>
      )}
      
      <div className="mt-auto text-left">
        <a
          href={linkHref}
          className="inline-flex items-center text-sm font-semibold text-brand-accent-subtle hover:text-brand-accent-highlight transition-colors duration-200 group/link"
          aria-label={t(`productModalitiesSection.${ariaLabelKeySuffix}.linkLabel`)}
        >
          {linkLabel}
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 ml-1.5 transform transition-transform group-hover/link:translate-x-1 duration-200">
            <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
          </svg>
        </a>
      </div>
    </AnimatedElement>
  );
};


export const ProductModalitiesSection: React.FC = () => {
  const { t, tArray } = useLanguage();

  const hatModalities = [
    {
      id: 'spotway',
      title: t('productModalitiesSection.spotway.title'),
      description: t('productModalitiesSection.spotway.description'),
      logoElement: <LogoSpotWay className="h-12 md:h-14 w-auto object-contain" altText={t('productModalitiesSection.spotway.title')} />,
      linkHref: '#spotway-option',
      linkLabel: t('productModalitiesSection.spotway.linkLabel'),
      tags: tArray('productModalitiesSection.spotway.tags'),
      ariaLabelKeySuffix: 'spotway'
    },
    {
      id: 'raspway',
      title: t('productModalitiesSection.raspway.title'),
      description: t('productModalitiesSection.raspway.description'),
      logoElement: <LogoRaspWay className="h-12 md:h-14 w-auto object-contain" altText={t('productModalitiesSection.raspway.title')} />,
      linkHref: '#raspway-option',
      linkLabel: t('productModalitiesSection.raspway.linkLabel'),
      tags: tArray('productModalitiesSection.raspway.tags'),
      ariaLabelKeySuffix: 'raspway'
    }
  ];

  const mobileModality = {
    id: 'mobile',
    title: t('productModalitiesSection.mobileApp.title'),
    description: t('productModalitiesSection.mobileApp.description'),
    logoElement: <IconMobileApp className="w-12 h-12 md:w-14 md:h-14 text-brand-accent-subtle" />,
    linkHref: '#trackie-mobile',
    linkLabel: t('productModalitiesSection.mobileApp.linkLabel'),
    tags: tArray('productModalitiesSection.mobileApp.tags'),
    ariaLabelKeySuffix: 'mobileApp'
  };

  return (
    <section id="product-modalities" className="py-28 md:py-40 bg-brand-surface/30 content-over-noise">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('productModalitiesSection.subtitle')}
          title={t('productModalitiesSection.title')}
          description={t('productModalitiesSection.description')}
        />

        <AnimatedElement initialClasses="opacity-0" finalClasses="opacity-100" transitionClasses="transition-opacity duration-1000 ease-in delay-300ms">
          <h3 className="text-3xl md:text-4xl font-bold text-brand-primary-text mt-16 mb-12 text-center md:text-left" style={{textShadow: '0 0 12px rgba(var(--color-brand-accent-glow-rgb), 0.4)'}}>
            {t('productModalitiesSection.hatTitle')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-10 mb-20">
            {hatModalities.map((modality, index) => (
              <ModalityCard key={modality.id} {...modality} index={index} />
            ))}
          </div>
        </AnimatedElement>

        <AnimatedElement initialClasses="opacity-0" finalClasses="opacity-100" transitionClasses="transition-opacity duration-1000 ease-in delay-500ms">
          <h3 className="text-3xl md:text-4xl font-bold text-brand-primary-text mt-16 mb-12 text-center md:text-left" style={{textShadow: '0 0 12px rgba(var(--color-brand-accent-subtle-rgb), 0.4)'}}>
            {t('productModalitiesSection.mobileTitle')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-1">
             <div className="max-w-lg mx-auto w-full">
                <ModalityCard {...mobileModality} index={0} />
             </div>
          </div>
        </AnimatedElement>

      </div>
    </section>
  );
};
