
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
import { useLanguage } from '../i18n/LanguageContext';

// Example icons (replace with actual, more abstract/benefit-oriented icons if available)
const IconAccessibility = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3" /></svg>;
const IconSafetyShield = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" /></svg>;
const IconNextGenAssist = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.354a15.054 15.054 0 01-4.5 0M12 6.75h.008v.008H12V6.75zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" /></svg>;
const IconIndustry = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 21h16.5M4.5 3h15M5.25 3v18m13.5-18v18M9 6.75h6.375M9 12h6.375m-6.375 5.25h6.375M5.25 9h1.5m3.75 0h1.5m3.75 0h1.5m-9.75 8.25h1.5m3.75 0h1.5m3.75 0h1.5M7.5 21V3m9 18V3" /></svg>;

// Define metadata for benefits, icons remain as components.
const benefitsMeta = [
  { id: 'b1', icon: <IconAccessibility />, titleKey: 'benefitsSection.items.0.title', textKey: 'benefitsSection.items.0.text' },
  { id: 'b2', icon: <IconSafetyShield />, titleKey: 'benefitsSection.items.1.title', textKey: 'benefitsSection.items.1.text' },
  { id: 'b3', icon: <IconNextGenAssist />, titleKey: 'benefitsSection.items.2.title', textKey: 'benefitsSection.items.2.text' },
  { id: 'b4', icon: <IconIndustry />, titleKey: 'benefitsSection.items.3.title', textKey: 'benefitsSection.items.3.text' },
];


export const BenefitsSection: React.FC = () => {
  const { t } = useLanguage();
  return (
    <section id="benefits" className="py-28 md:py-40 bg-brand-bg content-over-noise">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('benefitsSection.subtitle')}
          title={t('benefitsSection.title')}
          align="left"
          description={t('benefitsSection.description')}
        />
        <div className="mt-16 md:mt-20 grid md:grid-cols-2 gap-x-10 lg:gap-x-12 gap-y-10 md:gap-y-12">
          {benefitsMeta.map((benefitMeta, index) => (
            <AnimatedElement 
              key={benefitMeta.id} 
              initialClasses="opacity-0 translate-y-10"
              finalClasses="opacity-100 translate-y-0"
              transitionClasses={`transition-all duration-700 ease-out`}
              style={{transitionDelay: `${index * 150 + 300}ms`}}
              className="group p-6 md:p-8 rounded-xl bg-brand-surface/50 backdrop-blur-md hover:bg-brand-surface-alt/60 border border-brand-surface-alt/50 hover:border-brand-accent-subtle/40 transition-all duration-300 shadow-xl hover:shadow-brand-accent-glow/10 transform hover:-translate-y-1"
            >
              {benefitMeta.icon && (
                <div className="mb-4 text-brand-accent-glow">
                  {React.cloneElement(benefitMeta.icon, { className: "w-8 h-8" })}
                </div>
              )}
              <h3 className="text-xl md:text-2xl font-bold text-brand-primary-text mb-3 group-hover:text-brand-accent-highlight transition-colors duration-200">{t(benefitMeta.titleKey)}</h3>
              <p className="text-md text-brand-secondary-text leading-relaxed">{t(benefitMeta.textKey)}</p>
            </AnimatedElement>
          ))}
        </div>
      </div>
    </section>
  );
};
