import React, { useEffect, useRef, useState } from 'react';

const WrappedFMLanding = () => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [isHovered, setIsHovered] = useState(false);

  // Data points for sparkline
  const sparklinePoints = useRef(
    Array.from({ length: 50 }, (_, i) => {
      const x = (i / 49) * 200;
      const y = 50 + Math.sin(i * 0.2) * 20 + Math.random() * 10;
      return { x, y };
    })
  );

  // Sparkline animation
  const sparklineDot = useRef({ index: 0, startTime: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    let blobs = [];
    let particles = [];
    let initialized = false;
    let time = 0;

    const initializeElements = () => {
      if (canvas.width > 0 && canvas.height > 0 && !initialized) {
        blobs = Array.from({ length: 5 }, (_, i) => ({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          radius: 150 + Math.random() * 200,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          hue: 240 + (i * 30),
          saturation: 60 + Math.random() * 40,
          lightness: 40 + Math.random() * 20,
          opacity: 0.1 + Math.random() * 0.1,
          phase: Math.random() * Math.PI * 2
        }));

        particles = Array.from({ length: 40 }, (_, i) => {
          let x, y;
          if (i < 20) {
            x = 50;
            y = 50;
          } else {
            x = canvas.width - 50;
            y = canvas.height - 50;
          }
          return {
            x,
            y,
            vx: (Math.random() - 0.5) * 0.2,
            vy: (Math.random() - 0.5) * 0.2,
            size: 1 + Math.random() * 2,
            mass: 0.3 + Math.random() * 0.7,
            opacity: 0.3 + Math.random() * 0.4,
            pulse: Math.random() * Math.PI * 2
          };
        });

        initialized = true;
      }
    };

    const animate = (timestamp) => {
      if (!initialized) initializeElements();
      if (!initialized || blobs.length === 0 || particles.length === 0) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }

      time += 0.01;
      ctx.fillStyle = 'rgba(15, 15, 35, 1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      blobs.forEach((blob, index) => {
        blob.x += blob.vx + Math.sin(time + blob.phase) * 0.2;
        blob.y += blob.vy + Math.cos(time + blob.phase * 1.5) * 0.15;

        if (blob.x < -blob.radius/2) blob.x = canvas.width + blob.radius/2;
        if (blob.x > canvas.width + blob.radius/2) blob.x = -blob.radius/2;
        if (blob.y < -blob.radius/2) blob.y = canvas.height + blob.radius/2;
        if (blob.y > canvas.height + blob.radius/2) blob.y = -blob.radius/2;

        const gradient = ctx.createRadialGradient(blob.x, blob.y, 0, blob.x, blob.y, blob.radius);
        const dynamicOpacity = blob.opacity + Math.sin(time + index) * 0.05;
        gradient.addColorStop(0, `hsla(${blob.hue}, ${blob.saturation}%, ${blob.lightness}%, ${dynamicOpacity})`);
        gradient.addColorStop(0.6, `hsla(${blob.hue}, ${blob.saturation}%, ${blob.lightness}%, ${dynamicOpacity * 0.3})`);
        gradient.addColorStop(1, 'transparent');

        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(blob.x, blob.y, blob.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      particles.forEach((particle, i) => {
        let fx = 0, fy = 0;
        particles.forEach((other, j) => {
          if (i !== j) {
            const dx = particle.x - other.x;
            const dy = particle.y - other.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance > 0 && distance < 200) {
              const force = (particle.mass * other.mass) / (distance * distance) * 2.5;
              fx += (dx / distance) * force;
              fy += (dy / distance) * force;
            }
          }
        });
        particle.vx += fx;
        particle.vy += fy;
        particle.vx *= 0.995;
        particle.vy *= 0.995;

        const maxVel = 4.0;
        const vel = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
        if (vel > maxVel) {
          particle.vx = (particle.vx / vel) * maxVel;
          particle.vy = (particle.vy / vel) * maxVel;
        }

        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.pulse += 0.05;

        if (particle.x < 20) {
          particle.x = 20;
          particle.vx = Math.abs(particle.vx) * 0.5;
        }
        if (particle.x > canvas.width - 20) {
          particle.x = canvas.width - 20;
          particle.vx = -Math.abs(particle.vx) * 0.5;
        }
        if (particle.y < 20) {
          particle.y = 20;
          particle.vy = Math.abs(particle.vy) * 0.5;
        }
        if (particle.y > canvas.height - 20) {
          particle.y = canvas.height - 20;
          particle.vy = -Math.abs(particle.vy) * 0.5;
        }
      });

      ctx.globalCompositeOperation = 'source-over';
      particles.forEach(particle => {
        const pulseOpacity = particle.opacity + Math.sin(particle.pulse) * 0.2;
        ctx.fillStyle = `rgba(255, 255, 255, ${pulseOpacity})`;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Sparkline Dot (start after 5s)
      if (!sparklineDot.current.startTime && timestamp > 5000) {
        sparklineDot.current.startTime = timestamp;
      }

      if (sparklineDot.current.startTime) {
        const elapsed = timestamp - sparklineDot.current.startTime;
        const progress = Math.min(elapsed / 4000, 1);
        const index = Math.floor(progress * (sparklinePoints.current.length - 1));
        const trailLength = 10;

        for (let i = Math.max(0, index - trailLength); i <= index; i++) {
          const alpha = (i - (index - trailLength)) / trailLength;
          const { x, y } = sparklinePoints.current[i];
          ctx.beginPath();
          ctx.arc(canvas.width / 2 + x - 100, canvas.height * 0.85 + y - 50, 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.8})`;
          ctx.fill();
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full z-0" />;
};

export default WrappedFMLanding;
