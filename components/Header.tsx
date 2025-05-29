
import React, { useState, useEffect, useRef } from 'react';
import AnimatedElement from './AnimatedElement';
import { useLanguage } from '../i18n/LanguageContext';
import { NavItemLink } from '../types';
import { MobileNavPanel } from './MobileNavPanel';
import { LanguageSwitcher } from './LanguageSwitcher';

const GitHubIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
  </svg>
);

const HamburgerIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    </svg>
);


export const Header: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [openDropdownLabelKey, setOpenDropdownLabelKey] = useState<string | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const dropdownTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const { t } = useLanguage();

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 30);
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (dropdownTimeoutRef.current) clearTimeout(dropdownTimeoutRef.current);
    };
  }, []);

  const navItems: NavItemLink[] = [
    { href: "#what-is-trackie", labelKey: "header.whatIsTrackie" },
    { href: "#benefits", labelKey: "header.benefits" },
    {
      labelKey: "header.hatConcept",
      href: "#hat-concept", 
      subItems: [
        { href: "#spotway-option", labelKey: "header.spotway" },
        { href: "#raspway-option", labelKey: "header.raspway" },
      ],
    },
    { href: "#trackie-mobile", labelKey: "header.trackieMobile" },
    { href: "#features", labelKey: "header.features" },
    { href: "#ai-engine", labelKey: "header.aiEngine" },
    { href: "#vision", labelKey: "header.vision" },
  ];

  const handleMouseEnterDropdown = (labelKey: string) => {
    if (dropdownTimeoutRef.current) clearTimeout(dropdownTimeoutRef.current);
    setOpenDropdownLabelKey(labelKey);
  };

  const handleMouseLeaveDropdown = () => {
    dropdownTimeoutRef.current = setTimeout(() => setOpenDropdownLabelKey(null), 200);
  };
  
  const handleSubItemClick = () => {
    if (dropdownTimeoutRef.current) clearTimeout(dropdownTimeoutRef.current);
    setOpenDropdownLabelKey(null);
  };

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);

  return (
    <>
      <header 
        className={`fixed top-0 left-0 right-0 z-30 transition-all duration-300 ease-in-out 
                    ${isScrolled ? 'bg-brand-bg/60 backdrop-blur-xl shadow-2xl' : 'bg-transparent'}`}
      >
        <div className="container mx-auto px-4 sm:px-6 lg:px-8"> {/* Reduced horizontal padding for smaller screens */}
          <div className="flex items-center justify-between h-16 md:h-20">
            <AnimatedElement initialClasses='opacity-0 -translate-x-5' finalClasses='opacity-100 translate-x-0' transitionClasses='transition-all duration-700 ease-out delay-100'>
              <a href="#root" className="text-2xl md:text-3xl font-bold text-brand-primary-text hover:text-brand-accent-highlight transition-colors duration-300">
                TrackWay
              </a>
            </AnimatedElement>
            
            {/* Desktop Navigation & Actions */}
            <div className="hidden md:flex items-center space-x-3 md:space-x-4">
              <AnimatedElement 
                className="relative z-20"
                initialClasses='opacity-0 translate-x-5' 
                finalClasses='opacity-100 translate-x-0' 
                transitionClasses='transition-all duration-700 ease-out delay-[200ms]'
              >
                <nav className="flex items-center space-x-5 lg:space-x-7">
                  {navItems.map((item) => (
                    item.subItems ? (
                      <div
                        key={item.labelKey}
                        className="relative group"
                        onMouseEnter={() => handleMouseEnterDropdown(item.labelKey)}
                        onMouseLeave={handleMouseLeaveDropdown}
                      >
                        <a
                          href={item.href}
                          className="relative z-10 text-[15px] font-medium text-brand-secondary-text hover:text-brand-accent-highlight transition-colors duration-200 flex items-center py-2.5"
                          aria-haspopup="true"
                          aria-expanded={openDropdownLabelKey === item.labelKey}
                        >
                          {t(item.labelKey)}
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={`w-4 h-4 ml-1.5 transition-transform duration-200 ${openDropdownLabelKey === item.labelKey ? 'transform rotate-180' : ''}`}>
                            <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.25 4.25a.75.75 0 01-1.06 0L5.23 8.29a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                          </svg>
                        </a>
                        <div
                          className={`absolute top-full left-1/2 -translate-x-1/2 mt-2 w-52 bg-brand-surface/90 backdrop-blur-lg shadow-2xl rounded-xl p-2.5 z-30 border border-brand-surface-alt/50
                                      transition-all ease-out duration-200 transform origin-top
                                      ${openDropdownLabelKey === item.labelKey ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}
                          role="menu"
                          aria-label={`${t(item.labelKey)} submenu`}
                        >
                          <ul className="space-y-1.5" role="none">
                            {item.subItems.map((subItem) => (
                              <li key={subItem.href} role="none">
                                <a
                                  href={subItem.href}
                                  className="block w-full text-left px-3.5 py-2.5 text-[14px] text-brand-secondary-text hover:text-brand-accent-highlight hover:bg-brand-surface-alt/70 rounded-md transition-all duration-200"
                                  onClick={handleSubItemClick}
                                  role="menuitem"
                                >
                                  {t(subItem.labelKey)}
                                </a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <a
                        key={item.href}
                        href={item.href}
                        className="relative z-10 text-[15px] font-medium text-brand-secondary-text hover:text-brand-accent-highlight transition-colors duration-200 py-2.5"
                      >
                        {t(item.labelKey)}
                      </a>
                    )
                  ))}
                </nav>
              </AnimatedElement>
              <AnimatedElement 
                className="relative z-10"
                initialClasses='opacity-0 translate-x-5' 
                finalClasses='opacity-100 translate-x-0' 
                transitionClasses='transition-all duration-700 ease-out delay-[300ms]'
              >
                <a
                  href="https://plataforma.gpinovacao.senai.br/plataforma/ideia/274056"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 text-sm font-semibold text-brand-bg bg-brand-accent-glow hover:bg-brand-accent-highlight border border-transparent rounded-lg shadow-lg shadow-brand-accent-glow/30 transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-brand-accent-glow/50 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand-accent-glow/40 group"
                >
                  {t('header.knowTheProject')}
                </a>
              </AnimatedElement>
              <AnimatedElement 
                  className="relative z-10"
                  initialClasses='opacity-0 translate-x-5' 
                  finalClasses='opacity-100 translate-x-0' 
                  transitionClasses='transition-all duration-700 ease-out delay-[400ms]'
                >
                <LanguageSwitcher />
              </AnimatedElement>
               <AnimatedElement 
                  className="relative z-10"
                  initialClasses='opacity-0 translate-x-5' 
                  finalClasses='opacity-100 translate-x-0' 
                  transitionClasses='transition-all duration-700 ease-out delay-[500ms]' // Increased delay
                >
                <a
                  href="#" // Replace with actual GitHub link
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t('header.githubAlt')}
                  className="flex items-center justify-center p-2.5 text-brand-secondary-text bg-brand-surface/70 backdrop-blur-md border border-brand-surface-alt/80 rounded-lg shadow-md transition-all duration-300 transform hover:scale-105 hover:text-brand-accent-highlight hover:bg-brand-surface-alt hover:border-brand-accent-highlight/60 hover:shadow-lg hover:shadow-brand-accent-highlight/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-highlight/50 group"
                >
                  <GitHubIcon className="w-5 h-5" />
                </a>
              </AnimatedElement>
            </div>

            {/* Mobile Menu Button & Language Switcher (visible on md and below) */}
            <div className="md:hidden flex items-center space-x-2">
                <AnimatedElement 
                    className="relative z-10"
                    initialClasses='opacity-0 translate-x-5' 
                    finalClasses='opacity-100 translate-x-0' 
                    transitionClasses='transition-all duration-700 ease-out delay-[200ms]'
                >
                    <button
                        onClick={toggleMobileMenu}
                        className="p-2.5 text-brand-secondary-text hover:text-brand-accent-highlight rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-glow"
                        aria-label={t('header.openMenu')}
                        aria-expanded={isMobileMenuOpen}
                        aria-controls="mobile-nav-panel"
                    >
                        <HamburgerIcon className="w-6 h-6" />
                    </button>
                </AnimatedElement>
            </div>
          </div>
        </div>
      </header>
      <MobileNavPanel isOpen={isMobileMenuOpen} onClose={toggleMobileMenu} navItems={navItems} />
    </>
  );
};
