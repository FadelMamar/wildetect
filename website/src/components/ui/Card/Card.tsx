import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  shadow?: boolean;
  to?: string;
}

export default function Card({ children, className, shadow = true, to }: CardProps) {
  const cardContent = (
    <div className={clsx(
      styles.card, 
      shadow && styles['card--shadow'],
      className
    )}>
      {children}
    </div>
  );

  if (to) {
    return (
      <Link to={to} className={styles.cardLink}>
        {cardContent}
      </Link>
    );
  }

  return cardContent;
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={clsx(styles.cardHeader, className)}>{children}</div>;
}

export function CardBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={clsx(styles.cardBody, className)}>{children}</div>;
}

export function CardFooter({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={clsx(styles.cardFooter, className)}>{children}</div>;
}
