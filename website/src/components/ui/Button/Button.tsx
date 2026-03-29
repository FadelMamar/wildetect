import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './Button.module.css';

interface ButtonProps {
  children: React.ReactNode;
  to?: string;
  href?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  className?: string;
  onClick?: () => void;
}

export default function Button({
  children,
  to,
  href,
  variant = 'primary',
  className,
  onClick,
}: ButtonProps) {
  const isLink = !!(to || href);
  const buttonClass = clsx(
    styles.button,
    styles[`button--${variant}`],
    className
  );

  if (isLink) {
    return (
      <Link to={to} href={href} className={buttonClass} onClick={onClick}>
        {children}
      </Link>
    );
  }

  return (
    <button className={buttonClass} onClick={onClick}>
      {children}
    </button>
  );
}
