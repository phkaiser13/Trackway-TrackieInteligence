
import React from 'react';
import { NavItemLink } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import { LanguageSwitcher } from './LanguageSwitcher';

interface MobileNavPanelProps {
  isOpen: boolean;
  onClose: () => void;
  navItems: NavItemLink[];
}

const GitHubIconSmall: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
  </svg>
);


export const MobileNavPanel: React.FC<MobileNavPanelProps> = ({ isOpen, onClose, navItems }) => {
  const { t } = useLanguage();
  const [openSubmenu, setOpenSubmenu] = React.useState<string | null>(null);

  const toggleSubmenu = (labelKey: string) => {
    setOpenSubmenu(openSubmenu === labelKey ? null : labelKey);
  };
  
  // Close panel when a main link (not submenu toggle) is clicked
  const handleLinkClick = () => {
    setOpenSubmenu(null); // Close any open submenus
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div 
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 transition-opacity duration-300 ease-in-out"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Panel */}
      <div 
        className={`fixed top-0 right-0 h-full w-72 sm:w-80 bg-brand-surface shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'} border-l border-brand-surface-alt`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="mobile-menu-title"
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-5 border-b border-brand-surface-alt">
            <h2 id="mobile-menu-title" className="text-lg font-semibold text-brand-primary-text">Menu</h2>
            <button 
              onClick={onClose} 
              className="p-2 text-brand-secondary-text hover:text-brand-accent-highlight rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-glow"
              aria-label={t('header.closeMenu')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <nav className="flex-grow p-5 space-y-2 overflow-y-auto">
            {navItems.map((item) => (
              <div key={item.labelKey}>
                {item.subItems ? (
                  <>
                    <button
                      onClick={() => toggleSubmenu(item.labelKey)}
                      className="w-full flex items-center justify-between text-left py-3 px-3 text-brand-primary-text hover:bg-brand-surface-alt rounded-md transition-colors duration-200"
                      aria-expanded={openSubmenu === item.labelKey}
                    >
                      {t(item.labelKey)}
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={`w-5 h-5 transition-transform duration-200 ${openSubmenu === item.labelKey ? 'rotate-180' : ''}`}>
                        <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.25 4.25a.75.75 0 01-1.06 0L5.23 8.29a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                      </svg>
                    </button>
                    {openSubmenu === item.labelKey && (
                      <ul className="pl-4 mt-1 space-y-1 border-l border-brand-surface-alt ml-2">
                        {item.subItems.map((subItem) => (
                          <li key={subItem.labelKey}>
                            <a
                              href={subItem.href}
                              onClick={handleLinkClick}
                              className="block py-2.5 px-3 text-brand-secondary-text hover:text-brand-accent-highlight hover:bg-brand-surface-alt rounded-md transition-colors duration-200"
                            >
                              {t(subItem.labelKey)}
                            </a>
                          </li>
                        ))}
                      </ul>
                    )}
                  </>
                ) : (
                  <a
                    href={item.href}
                    onClick={handleLinkClick}
                    className="block py-3 px-3 text-brand-primary-text hover:bg-brand-surface-alt rounded-md transition-colors duration-200"
                  >
                    {t(item.labelKey)}
                  </a>
                )}
              </div>
            ))}
             <div className="pt-4 mt-4 border-t border-brand-surface-alt space-y-3">
                <a
                    href="https://plataforma.gpinovacao.senai.br/plataforma/ideia/274056"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={handleLinkClick}
                    className="block py-3 px-3 text-brand-primary-text bg-brand-accent-glow/10 hover:bg-brand-accent-glow/20 rounded-md transition-colors duration-200 text-center font-medium"
                >
                    {t('header.knowTheProject')}
                </a>
                <a
                    href="#" // Replace with actual GitHub link
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={t('header.githubAlt')}
                    onClick={handleLinkClick}
                    className="flex items-center justify-center py-3 px-3 text-brand-secondary-text hover:text-brand-primary-text hover:bg-brand-surface-alt rounded-md transition-colors duration-200"
                >
                    <GitHubIconSmall className="w-5 h-5 mr-2" />
                    GitHub
                </a>
            </div>
          </nav>
          
          <div className="p-5 border-t border-brand-surface-alt">
            <LanguageSwitcher onSwitch={onClose} />
          </div>
        </div>
      </div>
    </>
  );
};
