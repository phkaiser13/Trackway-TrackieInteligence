
import React from 'react';
import { SectionTitle } from './SectionTitle';
import { FeatureCardGrok } from './FeatureCardGrok';
// FeatureType from types.ts is not directly used for the 'feature' prop of FeatureCardGrok
// as FeatureCardGrok expects already translated strings.
import { useLanguage } from '../i18n/LanguageContext';

const IconMultisensory = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 mb-3 text-brand-accent-glow"><path strokeLinecap="round" strokeLinejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V21m18-2.25V21M4.5 12A7.5 7.5 0 0112 4.5a7.5 7.5 0 017.5 7.5 7.5 7.5 0 01-7.5 7.5 7.5 7.5 0 01-7.5-7.5zm15 0a7.5 7.5 0 01-7.5 7.5M4.5 12a7.5 7.5 0 007.5 7.5m0-15a7.5 7.5 0 00-7.5 7.5m7.5-7.5V3m0 18V12m0 0h7.5m-7.5 0H3" /></svg>;
const IconFlexAI = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 mb-3 text-brand-accent-glow"><path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" /></svg>;
const IconDynamicMode = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 mb-3 text-brand-accent-glow"><path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.755A3.989 3.989 0 0112.755 9H21M3 12v-2.755A3.989 3.989 0 016.245 6H12M12 21v-2.755A3.989 3.989 0 0012.755 15H21M3 12h2.755A3.989 3.989 0 009 14.755V21m3-18h2.25m-.397 1.363A4.005 4.005 0 0118 9.516V12m-6-9v2.755m0 13.5V12" /></svg>;

// Metadata for features, icons remain as components.
// These are the "source" definitions with translation keys.
const featuresMeta = [
  {
    id: 'f1',
    icon: <IconMultisensory />,
    titleKey: 'featuresSection.items.0.title',
    descriptionKey: 'featuresSection.items.0.description',
    linkKey: 'featuresSection.items.0.link',
    imagePlaceholderClass: 'bg-gradient-to-br from-sky-700/50 to-blue-800/50',
    linkUrl: '#ai-engine',
  },
  {
    id: 'f2',
    icon: <IconFlexAI />,
    titleKey: 'featuresSection.items.1.title',
    descriptionKey: 'featuresSection.items.1.description',
    linkKey: 'featuresSection.items.1.link',
    imagePlaceholderClass: 'bg-gradient-to-br from-purple-700/50 to-fuchsia-800/50',
    linkUrl: '#ai-engine',
  },
  {
    id: 'f3',
    icon: <IconDynamicMode />,
    titleKey: 'featuresSection.items.2.title',
    descriptionKey: 'featuresSection.items.2.description',
    linkKey: 'featuresSection.items.2.link',
    imagePlaceholderClass: 'bg-gradient-to-br from-teal-700/50 to-green-800/50',
    linkUrl: '#ai-engine',
  },
   {
    id: 'f4',
    // No icon for this one by design
    titleKey: 'featuresSection.items.3.title',
    descriptionKey: 'featuresSection.items.3.description',
    imagePlaceholderClass: 'bg-gradient-to-br from-rose-700/50 to-red-800/50',
  },
  {
    id: 'f5',
    // No icon
    titleKey: 'featuresSection.items.4.title',
    descriptionKey: 'featuresSection.items.4.description',
    imagePlaceholderClass: 'bg-gradient-to-br from-amber-700/50 to-yellow-800/50',
  },
   {
    id: 'f6',
    // No icon
    titleKey: 'featuresSection.items.5.title',
    descriptionKey: 'featuresSection.items.5.description',
    imagePlaceholderClass: 'bg-gradient-to-br from-indigo-700/50 to-violet-800/50',
  },
];

export const FeaturesSection: React.FC = () => {
  const { t } = useLanguage();

  return (
    <section id="features" className="py-28 md:py-40 bg-brand-surface/30 content-over-noise">
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('featuresSection.subtitle')}
          title={t('featuresSection.title')}
          align="left"
          description={t('featuresSection.description')}
        />
        <div className="mt-16 md:mt-20 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 md:gap-10">
          {featuresMeta.map((meta, index) => {
            // Construct the object that FeatureCardGrok expects (with translated strings)
            const featureForGrok = {
              id: meta.id,
              icon: meta.icon, // Pass the icon component directly
              title: t(meta.titleKey),
              description: t(meta.descriptionKey),
              linkText: meta.linkKey ? t(meta.linkKey) : undefined,
              imagePlaceholderClass: meta.imagePlaceholderClass,
              linkUrl: meta.linkUrl,
            };
            // The type of 'featureForGrok' will be inferred by TypeScript
            // and should match FeatureCardGrokProps['feature']
            return <FeatureCardGrok key={featureForGrok.id} feature={featureForGrok} index={index} />;
          })}
        </div>
      </div>
    </section>
  );
};
