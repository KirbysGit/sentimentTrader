import { Link } from "react-router-dom";
import { ArrowRight, BarChart3, Check, MessageCircle, TrendingUp, Zap } from "lucide-react";

import styled, { css } from "styled-components";

export function LandingPage() {
  return (
    <Page>
      <Nav>
        <NavInner>
          <Brand>
            <BrandIcon>
              <TrendingUp size={18} color="#ffffff" />
            </BrandIcon>
            <BrandText>Reddit Stock Insights</BrandText>
          </Brand>

          <NavActions>
            <TextLink as={Link} to="/login">
              Log in
            </TextLink>
            <ButtonLink as={Link} to="/signup" $variant="primary">
              Get Started
            </ButtonLink>
          </NavActions>
        </NavInner>
      </Nav>

      <Hero>
        <HeroInner>
          <Pill>
            <Zap size={16} />
            AI-Powered Social Sentiment Analysis
          </Pill>

          <H1>
            Turn Reddit Buzz into <GradientText>Actionable Insights</GradientText>
          </H1>

          <Subtitle>
            Track stock sentiment across Reddit communities. Discover trending tickers, monitor sentiment shifts, and
            generate signals from your pipeline.
          </Subtitle>

          <CtaRow>
            <ButtonLink as={Link} to="/signup" $variant="primary">
              Start Free Trial <ArrowRight size={18} />
            </ButtonLink>
            <ButtonLink as={Link} to="/login" $variant="secondary">
              View Demo
            </ButtonLink>
          </CtaRow>
        </HeroInner>
      </Hero>

      <Section>
        <Grid3>
          <FeatureCard
            icon={<MessageCircle size={28} />}
            title="Reddit Sentiment Tracking"
            description="Monitor daily discussions across selected subreddits and extract tickers with context."
            accent="#2563eb"
          />
          <FeatureCard
            icon={<Zap size={28} />}
            title="AI-Generated Signals"
            description="Use sentiment + engagement + price context to train models and generate signal probabilities."
            accent="#7c3aed"
          />
          <FeatureCard
            icon={<BarChart3 size={28} />}
            title="Trend Analytics"
            description="See sentiment shifts and simple trend context to spot narratives early."
            accent="#f97316"
          />
        </Grid3>
      </Section>

      <Section>
        <Proof>
          <Grid3>
            <StatItem label="Daily Mentions Analyzed" value="15K+" color="#2563eb" />
            <StatItem label="Signal Accuracy" value="92%" color="#7c3aed" />
            <StatItem label="Subreddits Monitored" value="50+" color="#16a34a" />
          </Grid3>
        </Proof>
      </Section>

      <Section>
        <Cta>
          <H2>Ready to outsmart the market?</H2>
          <CtaSub>Build your own sentiment-driven workflow: collect → score → merge → train → predict.</CtaSub>

          <Bullets>
            <BulletItem text="7-day free trial" />
            <BulletItem text="No credit card required" />
            <BulletItem text="Cancel anytime" />
          </Bullets>

          <ButtonLink as={Link} to="/signup" $variant="secondary">
            Get Started Free <ArrowRight size={18} />
          </ButtonLink>
        </Cta>
      </Section>

      <Footer>
        <FooterInner>© 2026 Reddit Stock Insights. All rights reserved.</FooterInner>
      </Footer>
    </Page>
  );
}

function FeatureCard({ icon, title, description, accent }) {
  return (
    <Card>
      <FeatureIcon $accent={accent}>{icon}</FeatureIcon>
      <H3>{title}</H3>
      <P>{description}</P>
    </Card>
  );
}

function BulletItem({ text }) {
  return (
    <Bullet>
      <Check size={18} />
      <span>{text}</span>
    </Bullet>
  );
}

function StatItem({ label, value, color }) {
  return (
    <Stat>
      <StatValue $color={color}>{value}</StatValue>
      <StatLabel>{label}</StatLabel>
    </Stat>
  );
}

const Page = styled.div`
  min-height: 100vh;
  color: ${({ theme }) => theme.colors.ink};
  background: ${({ theme }) => theme.gradients.page};
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
    "Apple Color Emoji", "Segoe UI Emoji";
`;

const Nav = styled.nav`
  position: sticky;
  top: 0;
  z-index: 10;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.glass};
  backdrop-filter: blur(10px);
`;

const NavInner = styled.div`
  max-width: 1120px;
  margin: 0 auto;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
`;

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
`;

const BrandIcon = styled.div`
  width: 34px;
  height: 34px;
  border-radius: ${({ theme }) => theme.radius.sm}px;
  background: ${({ theme }) => theme.gradients.brand};
  display: grid;
  place-items: center;
`;

const BrandText = styled.span`
  font-size: 18px;
  font-weight: 800;
  color: ${({ theme }) => theme.colors.ink};
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const TextLink = styled.a`
  color: ${({ theme }) => theme.colors.slate};
  text-decoration: none;
  font-weight: 600;
`;

const ButtonLink = styled.a`
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: ${({ theme }) => theme.radius.md}px;
  font-weight: 800;
  border: 1px solid transparent;

  ${({ $variant, theme }) =>
    $variant === "primary" &&
    css`
      background: ${theme.gradients.brand};
      color: #fff;
      box-shadow: ${theme.shadow.primary};
    `}

  ${({ $variant, theme }) =>
    $variant === "secondary" &&
    css`
      background: ${theme.colors.white};
      color: ${theme.colors.slate};
      border-color: rgba(15, 23, 42, 0.18);
    `}

  ${({ $variant }) =>
    $variant === "ghost" &&
    css`
      background: transparent;
      color: inherit;
      box-shadow: none;
      padding: 10px 12px;
    `}
`;

const Hero = styled.header`
  padding: 64px 16px 40px;
`;

const HeroInner = styled.div`
  max-width: 860px;
  margin: 0 auto;
  text-align: center;
`;

const Pill = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: rgba(124, 58, 237, 0.12);
  color: ${({ theme }) => theme.colors.primary};
  padding: 10px 14px;
  border-radius: ${({ theme }) => theme.radius.pill}px;
  font-weight: 700;
  font-size: 13px;
`;

const H1 = styled.h1`
  margin: 18px 0 12px;
  font-size: 52px;
  line-height: 1.05;
  letter-spacing: -1px;
`;

const GradientText = styled.span`
  background: ${({ theme }) => theme.gradients.brand};
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
`;

const Subtitle = styled.p`
  margin: 0 auto;
  max-width: 720px;
  font-size: 18px;
  color: ${({ theme }) => theme.colors.muted};
  line-height: 1.6;
`;

const CtaRow = styled.div`
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
`;

const Section = styled.section`
  padding: 28px 16px;
`;

const Grid3 = styled.div`
  max-width: 1120px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;

  @media (max-width: 900px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background: ${({ theme }) => theme.colors.white};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.lg}px;
  padding: 18px;
  box-shadow: ${({ theme }) => theme.shadow.soft};
`;

const FeatureIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: ${({ theme }) => theme.radius.md}px;
  color: #ffffff;
  display: grid;
  place-items: center;
  margin-bottom: 12px;
  background: ${({ $accent, theme }) => $accent || theme.gradients.brand};
`;

const H3 = styled.h3`
  margin: 0 0 6px;
  font-size: 18px;
`;

const P = styled.p`
  margin: 0;
  color: #475569;
  line-height: 1.5;
`;

const Proof = styled.div`
  max-width: 1120px;
  margin: 0 auto;
  background: ${({ theme }) => theme.colors.white};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.xl}px;
  padding: 22px;
`;

const Stat = styled.div`
  text-align: center;
  padding: 10px;
`;

const StatValue = styled.div`
  font-size: 34px;
  font-weight: 900;
  margin-bottom: 6px;
  color: ${({ $color, theme }) => $color || theme.colors.primary};
`;

const StatLabel = styled.div`
  color: #475569;
  font-weight: 600;
`;

const Cta = styled.div`
  max-width: 1120px;
  margin: 0 auto;
  border-radius: ${({ theme }) => theme.radius.xl}px;
  padding: 26px;
  color: #ffffff;
  background: ${({ theme }) => theme.gradients.cta};
  text-align: center;
`;

const H2 = styled.h2`
  margin: 0 0 8px;
  font-size: 30px;
`;

const CtaSub = styled.p`
  margin: 0 auto 18px;
  max-width: 780px;
  font-size: 16px;
  color: rgba(255, 255, 255, 0.9);
`;

const Bullets = styled.div`
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 18px;
`;

const Bullet = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
`;

const Footer = styled.footer`
  margin-top: 24px;
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  background: ${({ theme }) => theme.colors.white};
`;

const FooterInner = styled.div`
  max-width: 1120px;
  margin: 0 auto;
  padding: 18px 16px;
  color: #64748b;
  text-align: center;
`;