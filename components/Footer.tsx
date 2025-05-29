
import React from 'react';
import { useLanguage } from '../i18n/LanguageContext';

export const Footer: React.FC = () => {
  const { t } = useLanguage();
  const currentYear = new Date().getFullYear();

  return (
    <footer className="py-12 md:py-16 bg-brand-bg border-t border-brand-surface/70 content-over-noise">
      <div className="container mx-auto px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center text-center md:text-left space-y-4 md:space-y-0">
          <p className="text-xs md:text-sm text-brand-secondary-text">
            {t('footer.copyright', { year: currentYear.toString() })}
            <span className="hidden sm:inline"> | </span>
            <br className="sm:hidden"/> 
            {t('footer.projectInfo')}
          </p>
          <div className="flex space-x-5">
             <a href="https://plataforma.gpinovacao.senai.br/plataforma/ideia/274056" target="_blank" rel="noopener noreferrer" className="text-xs md:text-sm text-brand-secondary-text hover:text-brand-accent-subtle transition-colors duration-200">
                {t('footer.projectLinkSenai')}
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
