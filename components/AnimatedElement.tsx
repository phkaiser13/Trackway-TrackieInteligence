
import React, { useEffect, useRef, useState, ReactNode } from 'react';

interface AnimatedElementProps {
  children: ReactNode;
  className?: string;
  initialClasses?: string;
  finalClasses?: string;
  transitionClasses?: string;
  threshold?: number;
  triggerOnce?: boolean;
  delayMs?: number; // For style-based delay if Tailwind `delay-*` isn't sufficient
  style?: React.CSSProperties; 
}

const AnimatedElement: React.FC<AnimatedElementProps> = ({
  children,
  className = '',
  initialClasses = 'opacity-0 translate-y-10', // Default initial state
  finalClasses = 'opacity-100 translate-y-0',   // Default final state
  transitionClasses = 'transition-all duration-700 ease-out', // Default transition
  threshold = 0.1, // Trigger when 10% of the element is visible
  triggerOnce = true,
  delayMs, // Use Tailwind delay classes (e.g., delay-200ms) in initialClasses or transitionClasses when possible
  style = {},
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const currentRef = domRef.current;
    if (!currentRef) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            if (triggerOnce) {
              observer.unobserve(currentRef);
            }
          } else if (!triggerOnce && isVisible) { // Only set to false if it was visible and triggerOnce is false
             // setIsVisible(false); // Typically, for scroll-out animations, you might re-apply initialClasses.
                                   // For simple fade-in-once, this line is often commented out.
          }
        });
      },
      { threshold }
    );

    observer.observe(currentRef);

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [threshold, triggerOnce, isVisible]); // Added isVisible to dependencies if re-triggering is needed.

  // Combine Tailwind classes and dynamic styles
  // Tailwind's delay classes (e.g., `delay-300`) should be preferred and applied in `transitionClasses` or `initialClasses`.
  // The `delayMs` prop is for cases where JavaScript-driven dynamic delay is absolutely necessary.
  const combinedStyle: React.CSSProperties = {
    ...style,
    ...(delayMs ? { transitionDelay: `${delayMs}ms`, animationDelay: `${delayMs}ms` } : {}),
  };
  
  return (
    <div
      ref={domRef}
      className={`${className} ${transitionClasses} ${isVisible ? finalClasses : initialClasses}`}
      style={combinedStyle}
    >
      {children}
    </div>
  );
};

export default AnimatedElement;
