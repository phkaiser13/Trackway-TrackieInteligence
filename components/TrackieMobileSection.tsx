
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { IconMobileApp } from '../constants'; 
import { useLanguage } from '../i18n/LanguageContext';

export const TrackieMobileSection: React.FC = () => {
  const { t, tArray } = useLanguage();
  const usageScenarios = tArray('trackieMobileSection.usageScenarios');

  const appStores = [
    { storeKey: 'trackieMobileSection.appStore', logoSrc: '../Logosymbols/app_store.png', altTextKey: 'trackieMobileSection.altAppStore', platform: 'iOS', soon: true, logoClass: "w-7 h-7 mr-2.5" },
    { storeKey: 'trackieMobileSection.googlePlay', logoSrc: '../Logosymbols/google_play_logo.png', altTextKey: 'trackieMobileSection.altGooglePlay', platform: 'Android', soon: true, logoClass: "w-7 h-7 mr-2.5" }
  ];
  
  return (
    <section id="trackie-mobile" className="py-28 md:py-40 bg-brand-bg content-over-noise">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('trackieMobileSection.subtitle')}
          title={t('trackieMobileSection.title')}
          align="left"
          description={t('trackieMobileSection.description')}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 md:gap-20 items-center mt-16 md:mt-20">
          <AnimatedElement 
            initialClasses="opacity-0 translate-x-[-30px]" 
            finalClasses="opacity-100 translate-x-0" 
            transitionClasses="transition-all duration-1000 ease-out delay-200ms"
          >
            <div className="space-y-6">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold text-brand-primary-text mb-5" style={{textShadow: '0 0 10px rgba(var(--color-brand-accent-subtle-rgb), 0.3)'}}>
                {t('trackieMobileSection.secondaryTitle')}
              </h3>
              <p 
                className="text-brand-secondary-text text-base md:text-lg leading-relaxed"
                dangerouslySetInnerHTML={{ __html: t('trackieMobileSection.secondaryDescription1')}}
              />
              <p className="text-brand-secondary-text text-base md:text-lg leading-relaxed">
                {t('trackieMobileSection.secondaryDescription2')}
              </p>
              <h4 className="text-xl md:text-2xl font-semibold text-brand-primary-text pt-5">{t('trackieMobileSection.usageScenariosTitle')}</h4>
              <ul className="list-disc list-inside text-brand-secondary-text text-[15px] space-y-2.5 pl-1">
                {usageScenarios.map((scenario, index) => (
                  <li key={index}>{scenario}</li>
                ))}
              </ul>
            </div>
          </AnimatedElement>
          
          <AnimatedElement 
            initialClasses="opacity-0 scale-90 lg:translate-x-[30px]" 
            finalClasses="opacity-100 scale-100 lg:translate-x-0" 
            transitionClasses="transition-all duration-1000 ease-out delay-400ms"
            className="group aspect-[9/16] md:aspect-auto md:h-[550px] lg:h-[600px] rounded-2xl overflow-hidden bg-brand-surface shadow-2xl p-3 md:p-4 border border-brand-surface-alt/70"
          >
            <div className="w-full h-full bg-brand-surface-alt rounded-lg flex items-center justify-center relative overflow-hidden">
              <img 
                src="/DEMONSTS/MOBILE_DEMONST.png" 
                alt={t('trackieMobileSection.title')} // Translating alt text
                className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity duration-500 group-hover:scale-105 transform"
                loading="lazy"
              />
               <IconMobileApp className="absolute w-20 h-20 md:w-24 md:h-24 text-brand-accent-glow/40 group-hover:text-brand-accent-glow/60 transition-all duration-300 transform group-hover:scale-110" />
               <div className="absolute inset-0 bg-gradient-to-t from-brand-bg/50 via-transparent to-transparent"></div>
            </div>
          </AnimatedElement>
        </div>

        <AnimatedElement
            initialClasses="opacity-0 translate-y-12"
            finalClasses="opacity-100 translate-y-0"
            transitionClasses="transition-all duration-1000 ease-out delay-600ms"
            className="mt-20 md:mt-28 text-center"
        >
            <h3 className="text-2xl md:text-3xl font-semibold text-brand-primary-text mb-10">
                {t('trackieMobileSection.experienceTitle')}
            </h3>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 md:gap-6">
                {appStores.map(item => (
                    <button
                        key={item.storeKey}
                        onClick={(e) => e.preventDefault()} 
                        disabled={item.soon}
                        className="group shine-effect relative w-full sm:w-auto min-w-[200px] flex items-center justify-center px-6 py-4 text-sm font-medium text-brand-primary-text rounded-xl overflow-hidden
                                   bg-brand-surface hover:bg-brand-surface-alt 
                                   border border-brand-surface-alt hover:border-brand-accent-subtle/70
                                   transition-all duration-300 ease-out
                                   focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-glow/50
                                   transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
                        aria-label={`${t(item.storeKey)}${item.soon ? ` - ${t('trackieMobileSection.comingSoon')}` : ""}`}
                    >
                        <img src={item.logoSrc} alt={t(item.altTextKey)} className={`${item.logoClass} filter ${item.soon ? 'grayscale' : ''}`} />
                        <div className="text-left">
                            <span className="block text-xs text-brand-secondary-text">{item.soon ? t('trackieMobileSection.comingSoon') : t('trackieMobileSection.availableOn')}</span>
                            <span className="block text-lg font-semibold">{t(item.storeKey)}</span>
                        </div>
                         {item.soon && <span className="absolute top-1.5 right-1.5 text-[10px] bg-yellow-500 text-yellow-900 px-1.5 py-0.5 rounded-full font-bold uppercase">{t('trackieMobileSection.comingSoon')}</span>}
                    </button>
                ))}
            </div>
        </AnimatedElement>
      </div>
    </section>
  );
};
