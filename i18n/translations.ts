
import { Language } from '../types';

type TranslationDict = {
  [key: string]: string | string[] | TranslationDict | TranslationDict[];
};

export const translations: Record<Language, TranslationDict> = {
  pt: {
    header: {
      whatIsTrackie: "O que é Trackie?",
      benefits: "Benefícios",
      hatConcept: "O Chapéu",
      spotway: "SpotWay",
      raspway: "RaspWay",
      trackieMobile: "Trackie Mobile",
      features: "Funcionalidades",
      aiEngine: "IA Engine",
      vision: "Visão",
      knowTheProject: "Conheça o Projeto",
      githubAlt: "Trackie no GitHub",
      language: "Idioma",
      closeMenu: "Fechar Menu",
      openMenu: "Abrir Menu",
    },
    hero: {
      title: "TrackWay",
      tagline: "Redefinindo a interação inteligente. Trackie combina uma IA fundamental, otimizada para assistência visual e apoiada por um banco de dados dedicado, com o poder de modelos avançados. Descubra seu assistente pessoal multimodal para um mundo mais acessível e conectado.",
      exploreButton: "Explorar Ecossistema",
      scrollDown: "Rolar para a próxima seção",
    },
    whatIsTrackieSection: {
      subtitle: "Sua Janela Inteligente para o Mundo",
      title: "Desvendando Trackie: Assistência Multimodal em Tempo Real",
      description: "Trackie é seu companheiro de IA proativo, projetado para ampliar sua percepção, promover autonomia e garantir segurança. Ele opera com uma base de IA proprietária, incluindo algoritmos treinados e um banco de dados otimizado para assistência visual. Esta fundação trabalha em sinergia com APIs e modelos de IA avançados de mercado (como Gemini 2.0 Flash e outros) para um processamento inteligente e interpretação ambiental profunda. De forma análoga à abordagem da Microsoft com o Copilot, onde uma IA fundamental é enriquecida por modelos mais amplos, Trackie vê, ouve e compreende o mundo ao seu redor para oferecer suporte inteligente e contextualizado no seu dia a dia.",
      secondaryTitle: "Mais que um App, Uma Nova Percepção",
      secondaryDescription1: "Trackie não é apenas um software; é uma nova forma de perceber e interagir com o ambiente. Funcionando em tempo real, ele combina sua IA fundamental com visão computacional avançada, processamento de áudio e a capacidade de modelos de IA de ponta para oferecer uma assistência verdadeiramente contextual e intuitiva.",
      secondaryDescription2: "Seu principal objetivo é capacitar o usuário, promovendo independência ao identificar obstáculos, ler textos, reconhecer rostos e objetos, e alertar sobre potenciais perigos. Trackie é seu par de olhos e ouvidos inteligentes, sempre atento para garantir sua segurança e bem-estar.",
      benefitPoints: {
        autonomy: {
          title: "Autonomia Ampliada",
          description: "Navegue com mais confiança, realize tarefas diárias com maior facilidade e explore novos ambientes com o suporte inteligente do Trackie."
        },
        safety: {
          title: "Segurança Proativa",
          description: "Detecta perigos como obstáculos inesperados, mudanças de nível e outros elementos que exigem atenção, oferecendo alertas discretos e oportunos."
        },
        interaction: {
          title: "Interação Natural e Contextual",
          description: "Compreende comandos de voz naturais e fornece feedback claro e conciso, tornando a interação tecnológica fluida e intuitiva."
        },
        perception: {
          title: "Percepção Multimodal Inteligente",
          description: "Sua IA fundamental e algoritmos especializados, em conjunto com a capacidade de modelos de IA externos, habilitam uma rica percepção multimodal. Isso permite ir além do áudio para ler placas, identificar produtos ou descrever cenas com notável precisão."
        }
      }
    },
    productModalitiesSection: {
      subtitle: "Versatilidade Trackie",
      title: "Múltiplas Formas de Interagir com o Mundo",
      description: "Trackie se adapta às suas necessidades, oferecendo desde a inteligência embarcada do Chapéu Inteligente com seus módulos dedicados, até a conveniência do aplicativo móvel.",
      hatTitle: "Com o Chapéu Inteligente TrackWay",
      mobileTitle: "Direto no seu Smartphone",
      spotway: {
        title: "Trackie - Módulo SpotWay",
        description: "O chapéu leve e ágil com sensores essenciais. Conecta-se ao seu smartphone via Bluetooth ou Wi-Fi, utilizando o poder de processamento do app Trackie Mobile para uma experiência rica e conectada.",
        linkLabel: "Conheça o SpotWay",
        tags: ['Leve', 'Conectado ao Smartphone', 'Custo-Benefício']
      },
      raspway: {
        title: "Trackie - Módulo RaspWay",
        description: "Máxima autonomia com processamento embarcado. Equipado com um Raspberry Pi, todas as funcionalidades da IA Trackie rodam diretamente no chapéu, ideal para independência total.",
        linkLabel: "Descubra o RaspWay",
        tags: ['Autônomo', 'Processamento Embarcado', 'Offline Robusto']
      },
      mobileApp: {
        title: "Trackie Mobile App",
        description: "A inteligência Trackie diretamente no seu smartphone. Utiliza a câmera e microfone do seu dispositivo para assistência visual e auditiva, tornando a IA avançada acessível e prática.",
        linkLabel: "Explore o App",
        tags: ['App Smartphone', 'Acessível', 'Usa Câmera Existente']
      }
    },
    benefitsSection: {
        subtitle: "Impacto Real e Foco de Mercado",
        title: "Revolucionando Setores, Empoderando Indivíduos",
        description: "Trackie transcende um simples assistente. É uma plataforma com potencial para transformar múltiplos setores, com um foco primário em acessibilidade e inclusão, mas com vastas aplicações B2B e B2C.",
        items: [
            {
                title: "Acessibilidade e Inclusão Ampliadas",
                text: "Para deficientes visuais: descrições ambientais, leitura de texto, identificação de obstáculos e pessoas, auxílio na navegação. Para idosos e pessoas com desafios cognitivos: suporte em tarefas, lembretes, e comunicação facilitada. O chapéu discreto promove independência sem estigma."
            },
            {
                title: "Segurança Pessoal e Ocupacional Reforçada",
                text: "Detecção de perigos como objetos cortantes, fogo, fios expostos (DANGER_CLASSES), e gases. No ambiente de trabalho, auxilia técnicos em campo, identificando equipamentos, fornecendo instruções hands-free e alertas em zonas de risco."
            },
            {
                title: "Assistência Pessoal de Próxima Geração",
                text: "Para qualquer indivíduo buscando um assistente mais inteligente, que compreenda visualmente o contexto para oferecer informações e realizar tarefas de forma mais eficaz que os assistentes de voz atuais."
            },
            {
                title: "Indústria 4.0 e Manufatura Inteligente",
                text: "Aplicações em controle de qualidade visual automatizado, orientação de montagem passo a passo para operadores e treinamento interativo imersivo em chão de fábrica."
            }
        ]
    },
    hatConceptSection: {
        subtitle: "O Coração da Inovação Vestível",
        title: "Chapéu TrackWay: Inteligência Discreta, Impacto Real",
        description: "Mais que um dispositivo, uma extensão natural das suas capacidades. O Chapéu TrackWay, em seus módulos SpotWay e RaspWay, integra tecnologia de ponta de forma discreta, incluindo sensores essenciais de segurança como detecção de fogo, gás e fumaça, seguindo a filosofia de design anti-segregação para derrubar barreiras.",
        secondaryTitle: "Tecnologia que se Molda a Você",
        secondaryDescription1: "O Chapéu TrackWay é um exemplo de como a tecnologia pode ser poderosa sem ser ostensiva. Cada componente, incluindo os sensores integrados de segurança ambiental, é cuidadosamente integrado para manter a aparência de um acessório de moda comum, garantindo que o usuário se sinta confiante e confortável.",
        secondaryDescription2: "Acreditamos que a verdadeira assistência vem da naturalidade. Por isso, nosso foco é em um design que promove a inclusão, permitindo que a tecnologia sirva ao usuário sem chamar atenção para si mesma.",
        featurePills: [
            "Câmeras HD Miniaturizadas",
            "Microfones com Redução de Ruído",
            "Sensores Ultrassônicos de Proximidade",
            "Alerta de Gás, Fogo e Fumaça Integrado"
        ],
        antiSegregationQuote: "<strong>Filosofia Anti-Segregação:</strong> Nosso compromisso é criar tecnologia que empodera sem diferenciar. O design do Trackie Hat é intencionalmente sutil para que seja usado com naturalidade em qualquer ambiente social ou profissional."
    },
    spotwaySection: {
        subtitle: "Trackie SpotWay",
        title: "Conectado à Inteligência do Seu Smartphone",
        description: "SpotWay é a configuração leve e ágil do Chapéu TrackWay, inteiramente baseado no ESP32. Ele opera em perfeita simbiose com o aplicativo Trackie Mobile no seu smartphone (Android/iOS), utilizando o poder de processamento do seu celular para uma experiência rica e conectada.",
        secondaryTitle: "Leveza no Chapéu, Poder no Bolso",
        secondaryDescription: "No SpotWay, o chapéu abriga os sensores essenciais e o microcontrolador ESP32 de baixo consumo. Este componente coleta dados do ambiente e os transmite de forma eficiente via Bluetooth ou Wi-Fi para o seu smartphone. É no aplicativo Trackie Mobile que a mágica da IA acontece, processando as informações e fornecendo feedback ao usuário.",
        detailCards: [
            { title: "Microcontrolador ESP32 Dedicado", description: "Uso exclusivo do ESP32 para gerenciamento de sensores e comunicação, mantendo o chapéu extremamente leve e com baixo consumo energético." },
            { title: "Processamento no App Mobile", description: "A maior parte da inteligência artificial e processamento de dados é executada pelo aplicativo Trackie no smartphone, aproveitando seu hardware robusto." },
            { title: "Custo-Benefício Otimizado", description: "Esta abordagem reduz a complexidade e o custo do hardware no chapéu, tornando a solução mais acessível sem comprometer a experiência principal." }
        ]
    },
    raspwaySection: {
        subtitle: "Trackie RaspWay",
        title: "Autonomia e Processamento Dedicado no Chapéu",
        description: "Para máxima independência e poder de processamento local, o RaspWay equipa o Chapéu TrackWay com um Raspberry Pi robusto. Esta configuração permite que todas as funcionalidades da IA Trackie rodem diretamente no dispositivo, ideal para quem busca uma solução totalmente autônoma.",
        secondaryTitle: "Inteligência Embarcada, Liberdade Total",
        secondaryDescription: "O RaspWay transforma o chapéu em um verdadeiro computador vestível. Com um Raspberry Pi (como o Model 4 ou mais recente) integrado, ele lida com todo o processamento de dados dos sensores, execução dos modelos de IA (incluindo os offline como Gemma) e interação com o usuário de forma independente.",
        highlightPoints: [
            { title: "Processamento Local Avançado", descriptionLines: ["Capacidade de rodar modelos de IA complexos diretamente no chapéu, garantindo respostas rápidas e funcionalidade completa mesmo sem conexão à internet ou a um smartphone."] },
            { title: "Independência Total", descriptionLines: ["Ideal para usuários que não possuem um smartphone compatível, preferem não depender de um segundo dispositivo, ou para cenários de uso onde a conectividade é limitada."] },
            { title: "Maior Robustez Offline", descriptionLines: ["Com processamento dedicado, as funcionalidades offline são expandidas, oferecendo uma experiência de assistência mais completa em qualquer situação.", "Pode incluir bateria de maior capacidade para suportar o processamento embarcado e otimizações térmicas."] }
        ]
    },
    trackieMobileSection: {
        subtitle: "Trackie Mobile",
        title: "Inteligência na Palma da Sua Mão",
        description: "O software Trackie também opera como um aplicativo independente em smartphones (Android/iOS), tornando a assistência inteligente ainda mais acessível e versátil.",
        secondaryTitle: "Acessibilidade e Conveniência",
        secondaryDescription1: "Esta modalidade do Trackie utiliza <strong>exclusivamente a câmera e o microfone embutidos no seu celular.</strong> Não há necessidade de sensores externos ou hardware adicional.",
        secondaryDescription2: "Aproveitando a tecnologia que a maioria das pessoas já carrega no bolso, o Trackie Mobile democratiza o acesso à IA avançada. É perfeito para quem busca uma solução leve, prática e sempre à mão.",
        usageScenariosTitle: "Cenários de Uso Sugeridos:",
        usageScenarios: [
            "Celular no bolso da camisa (com a câmera estrategicamente posicionada).",
            "Utilizado em um suporte de mesa para interações hands-free.",
            "Acoplado a um gimbal de mão para exploração e navegação assistida.",
            "Interações discretas via fones de ouvido com microfone."
        ],
        experienceTitle: "Experimente o Trackie Mobile",
        appStore: "App Store",
        googlePlay: "Google Play",
        comingSoon: "Em Breve",
        availableOn: "Disponível na",
        altAppStore: "Logo Apple App Store",
        altGooglePlay: "Logo Google Play Store",
    },
    featuresSection: {
      subtitle: "A Inteligência do Trackie",
      title: "Software Sofisticado, Interação Natural",
      description: "No cerne do Trackie, uma arquitetura de IA híbrida combina uma base de algoritmos treinados e um banco de dados especializado com a capacidade de modelos externos para processar informações de múltiplas fontes e interagir de forma inteligente, entregue de maneira acessível e inclusiva.",
      items: [
        {
          title: "Percepção Multissensorial Avançada",
          description: "Combina técnicas de Visão Computacional (YOLO, DeepFace, MiDaS) com sua IA fundamental e algoritmos de processamento de dados para identificação precisa de objetos, reconhecimento facial e estimativa de distância. A interação de voz natural é potencializada por esta fusão de tecnologias, resultando em uma compreensão contextual aprimorada.",
          link: "Saiba Mais"
        },
        {
          title: "Flexibilidade de Modelos de IA",
          description: "Trackie adota uma arquitetura de IA híbrida. Sua IA central, composta por algoritmos e um banco de dados otimizado para desafios de assistência visual, integra-se dinamicamente com modelos líderes como Gemini 2.0 Flash, GPT, Claude, e soluções open-source (Gemma, DeepSeek, Llama). O desenvolvimento ativo da IA proprietária JuliusKaiser visa aprofundar essa sinergia inteligente.",
          link: "Saiba Mais"
        },
        {
          title: "Modos de Operação Dinâmicos",
          description: "Online: acesso a APIs de ponta, informações em tempo real e capacidades completas. Offline: modelos locais (Gemma, etc.), trabalhando com a IA fundamental do Trackie, para funções cruciais de assistência visual e navegação.",
          link: "Saiba Mais"
        },
        {
          title: "Ecossistema de Ferramentas Trackie",
          description: "Mais de 51 ferramentas e micro-serviços, muitos desenvolvidos internamente e integrados à IA fundamental, habilitando funcionalidades específicas e otimizando a experiência via \"function calling\"."
        },
        {
          title: "Compatibilidade de Plataforma Ampla",
          description: "Foco primário em mobile (Android/iOS) e no chapéu, mas a arquitetura da IA (fundamental e expansível) permite execução em Windows, Linux e macOS, ampliando P&D e nichos."
        },
        {
          title: "Aprendizado Contínuo e Personalização",
          description: "Capacidade de aprender com interações, alimentar seu banco de dados com novos rostos e adaptar-se às preferências do usuário, tornando-se um assistente verdadeiramente pessoal."
        }
      ]
    },
    aiEngineSection: {
        subtitle: "O Coração da Inteligência",
        title: "Trackie AI Engine: Flexível, Poderoso, Adaptável",
        description: "A inteligência do Trackie reside em uma arquitetura flexível. Uma IA fundamental, com algoritmos e banco de dados próprios treinados para assistência visual, é a base que se expande ao integrar modelos de IA de ponta, tanto online quanto offline. Essa abordagem, semelhante ao conceito do Microsoft Copilot, permite ao Trackie ser agnóstico e utilizar a melhor ferramenta para cada cenário, enquanto desenvolve ativamente sua própria arquitetura otimizada, JuliusKaiser.",
        onlineTitle: "Modelos de IA Online de Ponta",
        offlineTitle: "Capacidade Offline Robusta",
        offlineDescription: "Essencial para garantir funcionalidade em qualquer lugar. Modelos como Gemma, DeepSeek e Llama são executados localmente no dispositivo, operando em conjunto com a IA e banco de dados fundamental do Trackie para manter as funções cruciais de assistência.",
        proprietaryTitle: "JuliusKaiser",
        proprietarySubtitle: "Em Breve",
        developedAt: "Desenvolvido no SENAI \"Ettore Zanini\"",
        waitNews: "Aguarde Novidades",
        knowMoreJulius: "Saiba mais sobre JuliusKaiser em breve",
        models: {
            gemini: {
                name: "Gemini 2.0 Flash",
                description: "Integrado para respostas naturais e contextualizadas, aproveitando a vanguarda da IA do Google para interações de voz e processamento multimodal, complementando a IA fundamental do Trackie.",
                features: ["Interação de voz natural", "Compreensão contextual avançada", "Processamento multimodal"]
            },
            openai: {
                name: "OpenAI Realtime API",
                description: "Foco em interações de baixíssima latência, ideal para respostas imediatas e diálogos fluidos em tempo real. Utiliza os modelos mais recentes da OpenAI otimizados para velocidade.",
                features: ["Processamento em tempo real", "Baixa latência", "Interações dinâmicas"]
            },
            namo: {
                name: "NAMO-SSLM",
                description: "Modelo de linguagem compacto e especializado em processamento de fala (Small Speech Language Model), otimizado para eficiência em interações verbais e comandos de voz.",
                features: ["Especializado em fala", "Compacto e eficiente", "Comandos de voz aprimorados"]
            },
            gemma: {
                name: "Gemma (Google)",
                description: "Modelo open-source leve e eficiente do Google, executado localmente para garantir funcionalidades cruciais de assistência visual e interação, em conjunto com a IA base do Trackie, mesmo sem internet.",
                features: ["Performance otimizada para edge", "Baseado em tecnologia Gemini", "Disponibilidade offline"]
            },
            deepseek: {
                name: "DeepSeek Coder",
                description: "Modelos open-source com forte capacidade em compreensão e geração de código, potencialmente usado para ferramentas internas e raciocínio lógico offline, integrado ao sistema Trackie.",
                features: ["Forte em lógica e código", "Comunidade ativa", "Execução local"]
            },
            llama: {
                name: "Llama Family (Meta)",
                description: "Família de modelos open-source da Meta, oferecendo diversas opções de tamanho e performance para processamento local, adaptando-se às capacidades do dispositivo e complementando a IA do Trackie.",
                features: ["Ampla gama de tamanhos", "Performance escalável", "Uso offline robusto"]
            },
            julius: {
                name: "JuliusKaiser",
                description: "Nossa arquitetura de IA superinteligente, proprietária, desenvolvida ativamente pela TrackWay no SENAI \"Ettore Zanini\". JuliusKaiser é a evolução da nossa IA fundamental, construída para trabalhar em harmonia com modelos externos. É focado na interpretação extrema de ambientes, compreensão profunda de cenários complexos, objetos e interações dinâmicas, visando eficiência, personalização e sinergia incomparáveis com nosso ecossistema.",
                features: ["Interpretação Ambiental Extrema", "Compreensão Avançada de Cenários", "Otimização para Hardware TrackWay", "Eficiência Energética Superior", "Personalização Profunda", "Evolução Contínua Focada no Usuário"]
            }
        },
        logoAlt: {
          openai: "Logo OpenAI",
          deepseek: "Logo DeepSeek",
          meta: "Logo Meta Llama",
          googleGemini: "Logo Google Gemini",
          googleGemma: "Logo Google Gemma",
          namoSSLM: "Logo NAMO-SSLM",
          juliusKaiser: "Logo JuliusKaiser AI",
          spotway: "Logo SpotWay Module",
          raspway: "Logo RaspWay Module"
        },
        modelTypes: {
          online: "Online",
          offline: "Offline",
          proprietary: "Proprietário"
        }
    },
    visionSection: {
        subtitle: "[ NOSSA VISÃO DE FUTURO ]",
        title: "Um Salto Quântico na Assistência Inteligente",
        description: "O Trackie, com seu ecossistema inteligente e sua arquitetura de IA híbrida – que une uma base de IA proprietária com algoritmos treinados e um banco de dados especializado, à flexibilidade de modelos de IA de ponta – representa um avanço na forma como a IA pode assistir e empoderar indivíduos. Projetado para ser um companheiro discreto, poderoso e acessível, oferece um novo nível de independência e interação com o mundo. Convidamos você a se juntar à TrackWay nesta jornada para tornar o futuro da assistência inteligente uma realidade para todos.",
        joinButton: "Junte-se à Revolução Trackie"
    },
    footer: {
        copyright: "TrackWay © {year}. Todos os direitos reservados.",
        projectInfo: "Projeto conceitual para demonstração.",
        projectLinkSenai: "Detalhes do Projeto (SENAI)"
    },
    general: {
        loading: "Carregando...",
    }
  },
  en: {
    header: {
      whatIsTrackie: "What is Trackie?",
      benefits: "Benefits",
      hatConcept: "The Hat",
      spotway: "SpotWay",
      raspway: "RaspWay",
      trackieMobile: "Trackie Mobile",
      features: "Features",
      aiEngine: "AI Engine",
      vision: "Vision",
      knowTheProject: "Know the Project",
      githubAlt: "Trackie on GitHub",
      language: "Language",
      closeMenu: "Close Menu",
      openMenu: "Open Menu",
    },
    hero: {
      title: "TrackWay",
      tagline: "Redefining intelligent interaction. Trackie combines foundational AI, optimized for visual assistance and supported by a dedicated database, with the power of advanced models. Discover your multimodal personal assistant for a more accessible and connected world.",
      exploreButton: "Explore Ecosystem",
      scrollDown: "Scroll to next section",
    },
    whatIsTrackieSection: {
      subtitle: "Your Smart Window to the World",
      title: "Unveiling Trackie: Real-Time Multimodal Assistance",
      description: "Trackie is your proactive AI companion, designed to broaden your perception, promote autonomy, and ensure safety. It operates with a proprietary AI foundation, including trained algorithms and a database optimized for visual assistance. This foundation works in synergy with advanced market AI APIs and models (like Gemini 2.0 Flash and others) for intelligent processing and deep environmental interpretation. Analogous to Microsoft's Copilot approach, where foundational AI is enriched by broader models, Trackie sees, hears, and understands the world around you to offer intelligent, contextualized support in your daily life.",
      secondaryTitle: "More Than an App, A New Perception",
      secondaryDescription1: "Trackie isn't just software; it's a new way to perceive and interact with the environment. Operating in real-time, it combines its foundational AI with advanced computer vision, audio processing, and the capability of cutting-edge AI models to offer truly contextual and intuitive assistance.",
      secondaryDescription2: "Its primary goal is to empower the user, promoting independence by identifying obstacles, reading texts, recognizing faces and objects, and alerting to potential dangers. Trackie is your pair of smart eyes and ears, always attentive to ensure your safety and well-being.",
      benefitPoints: {
        autonomy: {
          title: "Enhanced Autonomy",
          description: "Navigate more confidently, perform daily tasks with greater ease, and explore new environments with Trackie's intelligent support."
        },
        safety: {
          title: "Proactive Safety",
          description: "Detects hazards like unexpected obstacles, level changes, and other elements requiring attention, offering discreet and timely alerts."
        },
        interaction: {
          title: "Natural and Contextual Interaction",
          description: "Understands natural voice commands and provides clear, concise feedback, making technological interaction fluid and intuitive."
        },
        perception: {
          title: "Intelligent Multimodal Perception",
          description: "Its foundational AI and specialized algorithms, combined with the capability of external AI models, enable rich multimodal perception. This allows it to go beyond audio to read signs, identify products, or describe scenes with remarkable accuracy."
        }
      }
    },
    productModalitiesSection: {
      subtitle: "Trackie Versatility",
      title: "Multiple Ways to Interact with the World",
      description: "Trackie adapts to your needs, offering everything from the embedded intelligence of the Smart Hat with its dedicated modules, to the convenience of the mobile application.",
      hatTitle: "With the TrackWay Smart Hat",
      mobileTitle: "Directly on Your Smartphone",
      spotway: {
        title: "Trackie - SpotWay Module",
        description: "The lightweight and agile hat with essential sensors. It connects to your smartphone via Bluetooth or Wi-Fi, using the Trackie Mobile app's processing power for a rich, connected experience.",
        linkLabel: "Learn about SpotWay",
        tags: ['Lightweight', 'Smartphone Connected', 'Cost-Effective']
      },
      raspway: {
        title: "Trackie - RaspWay Module",
        description: "Maximum autonomy with on-board processing. Equipped with a Raspberry Pi, all Trackie AI functionalities run directly on the hat, ideal for total independence.",
        linkLabel: "Discover RaspWay",
        tags: ['Autonomous', 'Embedded Processing', 'Robust Offline']
      },
      mobileApp: {
        title: "Trackie Mobile App",
        description: "Trackie intelligence directly on your smartphone. It uses your device's camera and microphone for visual and auditory assistance, making advanced AI accessible and practical.",
        linkLabel: "Explore the App",
        tags: ['Smartphone App', 'Accessible', 'Uses Existing Camera']
      }
    },
     benefitsSection: {
        subtitle: "Real Impact and Market Focus",
        title: "Revolutionizing Sectors, Empowering Individuals",
        description: "Trackie transcends a simple assistant. It's a platform with the potential to transform multiple sectors, with a primary focus on accessibility and inclusion, but with vast B2B and B2C applications.",
        items: [
            {
                title: "Enhanced Accessibility and Inclusion",
                text: "For visually impaired: environmental descriptions, text reading, obstacle and person identification, navigation aid. For elderly and cognitively challenged: task support, reminders, and facilitated communication. The discreet hat promotes independence without stigma."
            },
            {
                title: "Reinforced Personal and Occupational Safety",
                text: "Detection of hazards like sharp objects, fire, exposed wires (DANGER_CLASSES), and gases. In the workplace, it assists field technicians by identifying equipment, providing hands-free instructions, and alerts in risk zones."
            },
            {
                title: "Next-Generation Personal Assistance",
                text: "For any individual seeking a smarter assistant that visually understands context to offer information and perform tasks more effectively than current voice assistants."
            },
            {
                title: "Industry 4.0 and Smart Manufacturing",
                text: "Applications in automated visual quality control, step-by-step assembly guidance for operators, and immersive interactive training on the factory floor."
            }
        ]
    },
    hatConceptSection: {
        subtitle: "The Heart of Wearable Innovation",
        title: "TrackWay Hat: Discreet Intelligence, Real Impact",
        description: "More than a device, a natural extension of your capabilities. The TrackWay Hat, in its SpotWay and RaspWay modules, integrates cutting-edge technology discreetly, including essential safety sensors for fire, gas, and smoke detection, following an anti-segregation design philosophy to break down barriers.",
        secondaryTitle: "Technology That Adapts to You",
        secondaryDescription1: "The TrackWay Hat is an example of how technology can be powerful without being obtrusive. Each component, including integrated environmental safety sensors, is carefully integrated to maintain the appearance of a common fashion accessory, ensuring the user feels confident and comfortable.",
        secondaryDescription2: "We believe true assistance comes from naturalness. Therefore, our focus is on a design that promotes inclusion, allowing technology to serve the user without drawing attention to itself.",
        featurePills: [
            "Miniaturized HD Cameras",
            "Noise-Cancelling Microphones",
            "Ultrasonic Proximity Sensors",
            "Integrated Gas, Fire, and Smoke Alert"
        ],
        antiSegregationQuote: "<strong>Anti-Segregation Philosophy:</strong> Our commitment is to create technology that empowers without differentiating. The Trackie Hat's design is intentionally subtle to be worn naturally in any social or professional environment."
    },
     spotwaySection: {
        subtitle: "Trackie SpotWay",
        title: "Connected to Your Smartphone's Intelligence",
        description: "SpotWay is the lightweight and agile configuration of the TrackWay Hat, entirely based on the ESP32. It operates in perfect symbiosis with the Trackie Mobile app on your smartphone (Android/iOS), utilizing your phone's processing power for a rich and connected experience.",
        secondaryTitle: "Lightness in the Hat, Power in Your Pocket",
        secondaryDescription: "In SpotWay, the hat houses essential sensors and the low-consumption ESP32 microcontroller. This component collects environmental data and efficiently transmits it via Bluetooth or Wi-Fi to your smartphone. The AI magic happens in the Trackie Mobile app, processing information and providing feedback to the user.",
        detailCards: [
            { title: "Dedicated ESP32 Microcontroller", description: "Exclusive use of ESP32 for sensor management and communication, keeping the hat extremely lightweight and energy-efficient." },
            { title: "Processing on Mobile App", description: "Most artificial intelligence and data processing are performed by the Trackie app on the smartphone, leveraging its robust hardware." },
            { title: "Optimized Cost-Effectiveness", description: "This approach reduces hardware complexity and cost in the hat, making the solution more accessible without compromising the core experience." }
        ]
    },
    raspwaySection: {
        subtitle: "Trackie RaspWay",
        title: "Autonomy and Dedicated On-Hat Processing",
        description: "For maximum independence and local processing power, RaspWay equips the TrackWay Hat with a robust Raspberry Pi. This configuration allows all Trackie AI functionalities to run directly on the device, ideal for those seeking a fully autonomous solution.",
        secondaryTitle: "Embedded Intelligence, Total Freedom",
        secondaryDescription: "RaspWay transforms the hat into a true wearable computer. With an integrated Raspberry Pi (like Model 4 or newer), it handles all sensor data processing, AI model execution (including offline ones like Gemma), and user interaction independently.",
        highlightPoints: [
            { title: "Advanced Local Processing", descriptionLines: ["Ability to run complex AI models directly on the hat, ensuring quick responses and full functionality even without internet or smartphone connection."] },
            { title: "Total Independence", descriptionLines: ["Ideal for users who don't have a compatible smartphone, prefer not to rely on a second device, or for usage scenarios with limited connectivity."] },
            { title: "Greater Offline Robustness", descriptionLines: ["With dedicated processing, offline functionalities are expanded, offering a more complete assistance experience in any situation.", "May include a larger capacity battery to support embedded processing and thermal optimizations."] }
        ]
    },
    trackieMobileSection: {
        subtitle: "Trackie Mobile",
        title: "Intelligence in the Palm of Your Hand",
        description: "Trackie software also operates as a standalone app on smartphones (Android/iOS), making intelligent assistance even more accessible and versatile.",
        secondaryTitle: "Accessibility and Convenience",
        secondaryDescription1: "This Trackie modality uses <strong>exclusively the built-in camera and microphone on your phone.</strong> No external sensors or additional hardware needed.",
        secondaryDescription2: "Leveraging the technology most people already carry, Trackie Mobile democratizes access to advanced AI. It's perfect for those seeking a lightweight, practical, and always-at-hand solution.",
        usageScenariosTitle: "Suggested Usage Scenarios:",
        usageScenarios: [
            "Phone in shirt pocket (with camera strategically positioned).",
            "Used on a desk stand for hands-free interactions.",
            "Attached to a handheld gimbal for assisted exploration and navigation.",
            "Discreet interactions via headphones with microphone."
        ],
        experienceTitle: "Try Trackie Mobile",
        appStore: "App Store",
        googlePlay: "Google Play",
        comingSoon: "Coming Soon",
        availableOn: "Available on",
        altAppStore: "Apple App Store logo",
        altGooglePlay: "Google Play Store logo",
    },
    featuresSection: {
      subtitle: "The Intelligence of Trackie",
      title: "Sophisticated Software, Natural Interaction",
      description: "At the core of Trackie, a hybrid AI architecture combines a trained algorithm base and a specialized database with the capability of external models to process information from multiple sources and interact intelligently, delivered in an accessible and inclusive manner.",
      items: [
        {
          title: "Advanced Multisensory Perception",
          description: "Combines Computer Vision techniques (YOLO, DeepFace, MiDaS) with its foundational AI and data processing algorithms for precise object identification, facial recognition, and distance estimation. Natural voice interaction is enhanced by this fusion of technologies, resulting in improved contextual understanding.",
          link: "Learn More"
        },
        {
          title: "AI Model Flexibility",
          description: "Trackie adopts a hybrid AI architecture. Its core AI, comprising algorithms and a database optimized for visual assistance challenges, dynamically integrates with leading models like Gemini 2.0 Flash, GPT, Claude, and open-source solutions (Gemma, DeepSeek, Llama). Active development of the proprietary JuliusKaiser AI aims to deepen this intelligent synergy.",
          link: "Learn More"
        },
        {
          title: "Dynamic Operating Modes",
          description: "Online: access to cutting-edge APIs, real-time information, and full capabilities. Offline: local models (Gemma, etc.), working with Trackie's foundational AI, for crucial visual assistance and navigation functions.",
          link: "Learn More"
        },
        {
          title: "Trackie Tool Ecosystem",
          description: "Over 51 tools and micro-services, many developed in-house and integrated with the foundational AI, enabling specific functionalities and optimizing the experience via \"function calling\"."
        },
        {
          title: "Broad Platform Compatibility",
          description: "Primary focus on mobile (Android/iOS) and the hat, but the AI architecture (foundational and expandable) allows execution on Windows, Linux, and macOS, broadening R&D and niches."
        },
        {
          title: "Continuous Learning and Personalization",
          description: "Ability to learn from interactions, feed its database with new faces, and adapt to user preferences, becoming a truly personal assistant."
        }
      ]
    },
    aiEngineSection: {
        subtitle: "The Heart of Intelligence",
        title: "Trackie AI Engine: Flexible, Powerful, Adaptable",
        description: "Trackie's intelligence lies in a flexible architecture. A foundational AI, with its own algorithms and database trained for visual assistance, is the base that expands by integrating cutting-edge AI models, both online and offline. This approach, similar to Microsoft's Copilot concept, allows Trackie to be agnostic and use the best tool for each scenario, while actively developing its own optimized architecture, JuliusKaiser.",
        onlineTitle: "Cutting-Edge Online AI Models",
        offlineTitle: "Robust Offline Capability",
        offlineDescription: "Essential to ensure functionality anywhere. Models like Gemma, DeepSeek, and Llama run locally on the device, operating in conjunction with Trackie's foundational AI and database to maintain crucial assistance functions.",
        proprietaryTitle: "JuliusKaiser",
        proprietarySubtitle: "Coming Soon",
        developedAt: "Developed at SENAI \"Ettore Zanini\"",
        waitNews: "Stay Tuned for News",
        knowMoreJulius: "Learn more about JuliusKaiser soon",
        models: {
            gemini: {
                name: "Gemini 2.0 Flash",
                description: "Integrated for natural and contextual responses, leveraging Google's cutting-edge AI for voice interactions and multimodal processing, complementing Trackie's foundational AI.",
                features: ["Natural voice interaction", "Advanced contextual understanding", "Multimodal processing"]
            },
            openai: {
                name: "OpenAI Realtime API",
                description: "Focus on ultra-low latency interactions, ideal for immediate responses and fluid real-time dialogues. Uses OpenAI's latest models optimized for speed.",
                features: ["Real-time processing", "Low latency", "Dynamic interactions"]
            },
            namo: {
                name: "NAMO-SSLM",
                description: "Compact language model specialized in speech processing (Small Speech Language Model), optimized for efficiency in verbal interactions and voice commands.",
                features: ["Specialized in speech", "Compact and efficient", "Enhanced voice commands"]
            },
            gemma: {
                name: "Gemma (Google)",
                description: "Lightweight and efficient open-source model from Google, executed locally to ensure crucial visual assistance and interaction functionalities, in conjunction with Trackie's base AI, even without internet.",
                features: ["Optimized performance for edge", "Based on Gemini technology", "Offline availability"]
            },
            deepseek: {
                name: "DeepSeek Coder",
                description: "Open-source models with strong capabilities in code understanding and generation, potentially used for internal tools and offline logical reasoning, integrated into the Trackie system.",
                features: ["Strong in logic and code", "Active community", "Local execution"]
            },
            llama: {
                name: "Llama Family (Meta)",
                description: "Family of open-source models from Meta, offering various size and performance options for local processing, adapting to device capabilities and complementing Trackie's AI.",
                features: ["Wide range of sizes", "Scalable performance", "Robust offline use"]
            },
            julius: {
                name: "JuliusKaiser",
                description: "Our proprietary, super-intelligent AI architecture, actively developed by TrackWay at SENAI \"Ettore Zanini\". JuliusKaiser is the evolution of our foundational AI, built to work harmoniously with external models. It focuses on extreme environmental interpretation, deep understanding of complex scenarios, objects, and dynamic interactions, aiming for unparalleled efficiency, personalization, and synergy with our ecosystem.",
                features: ["Extreme Environmental Interpretation", "Advanced Scenario Comprehension", "TrackWay Hardware Optimization", "Superior Energy Efficiency", "Deep Personalization", "Continuous User-Focused Evolution"]
            }
        },
        logoAlt: {
          openai: "OpenAI Logo",
          deepseek: "DeepSeek Logo",
          meta: "Meta Llama Logo",
          googleGemini: "Google Gemini Logo",
          googleGemma: "Google Gemma Logo",
          namoSSLM: "NAMO-SSLM Logo",
          juliusKaiser: "JuliusKaiser AI Logo",
          spotway: "SpotWay Module Logo",
          raspway: "RaspWay Module Logo"
        },
        modelTypes: {
          online: "Online",
          offline: "Offline",
          proprietary: "Proprietary"
        }
    },
    visionSection: {
        subtitle: "[ OUR FUTURE VISION ]",
        title: "A Quantum Leap in Intelligent Assistance",
        description: "Trackie, with its intelligent ecosystem and hybrid AI architecture – uniting a proprietary AI base with trained algorithms and a specialized database, with the flexibility of cutting-edge AI models – represents an advancement in how AI can assist and empower individuals. Designed to be a discreet, powerful, and accessible companion, it offers a new level of independence and interaction with the world. We invite you to join TrackWay on this journey to make the future of intelligent assistance a reality for everyone.",
        joinButton: "Join the Trackie Revolution"
    },
    footer: {
        copyright: "TrackWay © {year}. All rights reserved.",
        projectInfo: "Conceptual project for demonstration.",
        projectLinkSenai: "Project Details (SENAI)"
    },
    general: {
        loading: "Loading...",
    }
  },
  es: { // Placeholder - Copy from English or Portuguese and mark for translation
    header: {
      whatIsTrackie: "[ES] What is Trackie?",
      benefits: "[ES] Benefits",
      hatConcept: "[ES] The Hat",
      spotway: "[ES] SpotWay",
      raspway: "[ES] RaspWay",
      trackieMobile: "[ES] Trackie Mobile",
      features: "[ES] Features",
      aiEngine: "[ES] AI Engine",
      vision: "[ES] Vision",
      knowTheProject: "[ES] Know the Project",
      githubAlt: "[ES] Trackie on GitHub",
      language: "[ES] Language",
      closeMenu: "[ES] Close Menu",
      openMenu: "[ES] Open Menu",
    },
    hero: {
      title: "[ES] TrackWay",
      tagline: "[ES] Redefining intelligent interaction. Trackie combines foundational AI, optimized for visual assistance and supported by a dedicated database, with the power of advanced models. Discover your multimodal personal assistant for a more accessible and connected world.",
      exploreButton: "[ES] Explore Ecosystem",
      scrollDown: "[ES] Scroll to next section",
    },
    whatIsTrackieSection: {
      subtitle: "[ES] Your Smart Window to the World",
      title: "[ES] Unveiling Trackie: Real-Time Multimodal Assistance",
      description: "[ES] Trackie is your proactive AI companion, designed to broaden your perception, promote autonomy, and ensure safety. It operates with a proprietary AI foundation, including trained algorithms and a database optimized for visual assistance. This foundation works in synergy with advanced market AI APIs and models (like Gemini 2.0 Flash and others) for intelligent processing and deep environmental interpretation. Analogous to Microsoft's Copilot approach, where foundational AI is enriched by broader models, Trackie sees, hears, and understands the world around you to offer intelligent, contextualized support in your daily life.",
      secondaryTitle: "[ES] More Than an App, A New Perception",
      secondaryDescription1: "[ES] Trackie isn't just software; it's a new way to perceive and interact with the environment. Operating in real-time, it combines its foundational AI with advanced computer vision, audio processing, and the capability of cutting-edge AI models to offer truly contextual and intuitive assistance.",
      secondaryDescription2: "[ES] Its primary goal is to empower the user, promoting independence by identifying obstacles, reading texts, recognizing faces and objects, and alerting to potential dangers. Trackie is your pair of smart eyes and ears, always attentive to ensure your safety and well-being.",
      benefitPoints: {
        autonomy: {
          title: "[ES] Enhanced Autonomy",
          description: "[ES] Navigate more confidently, perform daily tasks with greater ease, and explore new environments with Trackie's intelligent support."
        },
        safety: {
          title: "[ES] Proactive Safety",
          description: "[ES] Detects hazards like unexpected obstacles, level changes, and other elements requiring attention, offering discreet and timely alerts."
        },
        interaction: {
          title: "[ES] Natural and Contextual Interaction",
          description: "[ES] Understands natural voice commands and provides clear, concise feedback, making technological interaction fluid and intuitive."
        },
        perception: {
          title: "[ES] Intelligent Multimodal Perception",
          description: "[ES] Its foundational AI and specialized algorithms, combined with the capability of external AI models, enable rich multimodal perception. This allows it to go beyond audio to read signs, identify products, or describe scenes with remarkable accuracy."
        }
      }
    },
    productModalitiesSection: {
      subtitle: "[ES] Trackie Versatility",
      title: "[ES] Multiple Ways to Interact with the World",
      description: "[ES] Trackie adapts to your needs, offering everything from the embedded intelligence of the Smart Hat with its dedicated modules, to the convenience of the mobile application.",
      hatTitle: "[ES] With the TrackWay Smart Hat",
      mobileTitle: "[ES] Directly on Your Smartphone",
      spotway: {
        title: "[ES] Trackie - SpotWay Module",
        description: "[ES] The lightweight and agile hat with essential sensors. It connects to your smartphone via Bluetooth or Wi-Fi, using the Trackie Mobile app's processing power for a rich, connected experience.",
        linkLabel: "[ES] Learn about SpotWay",
        tags: ['[ES] Lightweight', '[ES] Smartphone Connected', '[ES] Cost-Effective']
      },
      raspway: {
        title: "[ES] Trackie - RaspWay Module",
        description: "[ES] Maximum autonomy with on-board processing. Equipped with a Raspberry Pi, all Trackie AI functionalities run directly on the hat, ideal for total independence.",
        linkLabel: "[ES] Discover RaspWay",
        tags: ['[ES] Autonomous', '[ES] Embedded Processing', '[ES] Robust Offline']
      },
      mobileApp: {
        title: "[ES] Trackie Mobile App",
        description: "[ES] Trackie intelligence directly on your smartphone. It uses your device's camera and microphone for visual and auditory assistance, making advanced AI accessible and practical.",
        linkLabel: "[ES] Explore the App",
        tags: ['[ES] Smartphone App', '[ES] Accessible', '[ES] Uses Existing Camera']
      }
    },
    benefitsSection: {
        subtitle: "[ES] Real Impact and Market Focus",
        title: "[ES] Revolutionizing Sectors, Empowering Individuals",
        description: "[ES] Trackie transcends a simple assistant. It's a platform with the potential to transform multiple sectors, with a primary focus on accessibility and inclusion, but with vast B2B and B2C applications.",
        items: [
            {
                title: "[ES] Enhanced Accessibility and Inclusion",
                text: "[ES] For visually impaired: environmental descriptions, text reading, obstacle and person identification, navigation aid. For elderly and cognitively challenged: task support, reminders, and facilitated communication. The discreet hat promotes independence without stigma."
            },
            {
                title: "[ES] Reinforced Personal and Occupational Safety",
                text: "[ES] Detection of hazards like sharp objects, fire, exposed wires (DANGER_CLASSES), and gases. In the workplace, it assists field technicians by identifying equipment, providing hands-free instructions, and alerts in risk zones."
            },
            {
                title: "[ES] Next-Generation Personal Assistance",
                text: "[ES] For any individual seeking a smarter assistant that visually understands context to offer information and perform tasks more effectively than current voice assistants."
            },
            {
                title: "[ES] Industry 4.0 and Smart Manufacturing",
                text: "[ES] Applications in automated visual quality control, step-by-step assembly guidance for operators, and immersive interactive training on the factory floor."
            }
        ]
    },
    hatConceptSection: {
        subtitle: "[ES] The Heart of Wearable Innovation",
        title: "[ES] TrackWay Hat: Discreet Intelligence, Real Impact",
        description: "[ES] More than a device, a natural extension of your capabilities. The TrackWay Hat, in its SpotWay and RaspWay modules, integrates cutting-edge technology discreetly, including essential safety sensors for fire, gas, and smoke detection, following an anti-segregation design philosophy to break down barriers.",
        secondaryTitle: "[ES] Technology That Adapts to You",
        secondaryDescription1: "[ES] The TrackWay Hat is an example of how technology can be powerful without being obtrusive. Each component, including integrated environmental safety sensors, is carefully integrated to maintain the appearance of a common fashion accessory, ensuring the user feels confident and comfortable.",
        secondaryDescription2: "[ES] We believe true assistance comes from naturalness. Therefore, our focus is on a design that promotes inclusion, allowing technology to serve the user without drawing attention to itself.",
        featurePills: [
            "[ES] Miniaturized HD Cameras",
            "[ES] Noise-Cancelling Microphones",
            "[ES] Ultrasonic Proximity Sensors",
            "[ES] Integrated Gas, Fire, and Smoke Alert"
        ],
        antiSegregationQuote: "[ES] <strong>Anti-Segregation Philosophy:</strong> Our commitment is to create technology that empowers without differentiating. The Trackie Hat's design is intentionally subtle to be worn naturally in any social or professional environment."
    },
    spotwaySection: {
        subtitle: "[ES] Trackie SpotWay",
        title: "[ES] Connected to Your Smartphone's Intelligence",
        description: "[ES] SpotWay is the lightweight and agile configuration of the TrackWay Hat, entirely based on the ESP32. It operates in perfect symbiosis with the Trackie Mobile app on your smartphone (Android/iOS), utilizing your phone's processing power for a rich and connected experience.",
        secondaryTitle: "[ES] Lightness in the Hat, Power in Your Pocket",
        secondaryDescription: "[ES] In SpotWay, the hat houses essential sensors and the low-consumption ESP32 microcontroller. This component collects environmental data and efficiently transmits it via Bluetooth or Wi-Fi to your smartphone. The AI magic happens in the Trackie Mobile app, processing information and providing feedback to the user.",
        detailCards: [
            { title: "[ES] Dedicated ESP32 Microcontroller", description: "[ES] Exclusive use of ESP32 for sensor management and communication, keeping the hat extremely lightweight and energy-efficient." },
            { title: "[ES] Processing on Mobile App", description: "[ES] Most artificial intelligence and data processing are performed by the Trackie app on the smartphone, leveraging its robust hardware." },
            { title: "[ES] Optimized Cost-Effectiveness", description: "[ES] This approach reduces hardware complexity and cost in the hat, making the solution more accessible without compromising the core experience." }
        ]
    },
    raspwaySection: {
        subtitle: "[ES] Trackie RaspWay",
        title: "[ES] Autonomy and Dedicated On-Hat Processing",
        description: "[ES] For maximum independence and local processing power, RaspWay equips the TrackWay Hat with a robust Raspberry Pi. This configuration allows all Trackie AI functionalities to run directly on the device, ideal for those seeking a fully autonomous solution.",
        secondaryTitle: "[ES] Embedded Intelligence, Total Freedom",
        secondaryDescription: "[ES] RaspWay transforms the hat into a true wearable computer. With an integrated Raspberry Pi (like Model 4 or newer), it handles all sensor data processing, AI model execution (including offline ones like Gemma), and user interaction independently.",
        highlightPoints: [
            { title: "[ES] Advanced Local Processing", descriptionLines: ["[ES] Ability to run complex AI models directly on the hat, ensuring quick responses and full functionality even without internet or smartphone connection."] },
            { title: "[ES] Total Independence", descriptionLines: ["[ES] Ideal for users who don't have a compatible smartphone, prefer not to rely on a second device, or for usage scenarios with limited connectivity."] },
            { title: "[ES] Greater Offline Robustness", descriptionLines: ["[ES] With dedicated processing, offline functionalities are expanded, offering a more complete assistance experience in any situation.", "[ES] May include a larger capacity battery to support embedded processing and thermal optimizations."] }
        ]
    },
    trackieMobileSection: {
        subtitle: "[ES] Trackie Mobile",
        title: "[ES] Intelligence in the Palm of Your Hand",
        description: "[ES] Trackie software also operates as a standalone app on smartphones (Android/iOS), making intelligent assistance even more accessible and versatile.",
        secondaryTitle: "[ES] Accessibility and Convenience",
        secondaryDescription1: "[ES] This Trackie modality uses <strong>exclusively the built-in camera and microphone on your phone.</strong> No external sensors or additional hardware needed.",
        secondaryDescription2: "[ES] Leveraging the technology most people already carry, Trackie Mobile democratizes access to advanced AI. It's perfect for those seeking a lightweight, practical, and always-at-hand solution.",
        usageScenariosTitle: "[ES] Suggested Usage Scenarios:",
        usageScenarios: [
            "[ES] Phone in shirt pocket (with camera strategically positioned).",
            "[ES] Used on a desk stand for hands-free interactions.",
            "[ES] Attached to a handheld gimbal for assisted exploration and navigation.",
            "[ES] Discreet interactions via headphones with microphone."
        ],
        experienceTitle: "[ES] Try Trackie Mobile",
        appStore: "[ES] App Store",
        googlePlay: "[ES] Google Play",
        comingSoon: "[ES] Coming Soon",
        availableOn: "[ES] Available on",
        altAppStore: "[ES] Apple App Store logo",
        altGooglePlay: "[ES] Google Play Store logo",
    },
    featuresSection: {
      subtitle: "[ES] The Intelligence of Trackie",
      title: "[ES] Sophisticated Software, Natural Interaction",
      description: "[ES] At the core of Trackie, a hybrid AI architecture combines a trained algorithm base and a specialized database with the capability of external models to process information from multiple sources and interact intelligently, delivered in an accessible and inclusive manner.",
      items: [
        {
          title: "[ES] Advanced Multisensory Perception",
          description: "[ES] Combines Computer Vision techniques (YOLO, DeepFace, MiDaS) with its foundational AI and data processing algorithms for precise object identification, facial recognition, and distance estimation. Natural voice interaction is enhanced by this fusion of technologies, resulting in improved contextual understanding.",
          link: "[ES] Learn More"
        },
        {
          title: "[ES] AI Model Flexibility",
          description: "[ES] Trackie adopts a hybrid AI architecture. Its core AI, comprising algorithms and a database optimized for visual assistance challenges, dynamically integrates with leading models like Gemini 2.0 Flash, GPT, Claude, and open-source solutions (Gemma, DeepSeek, Llama). Active development of the proprietary JuliusKaiser AI aims to deepen this intelligent synergy.",
          link: "[ES] Learn More"
        },
        {
          title: "[ES] Dynamic Operating Modes",
          description: "[ES] Online: access to cutting-edge APIs, real-time information, and full capabilities. Offline: local models (Gemma, etc.), working with Trackie's foundational AI, for crucial visual assistance and navigation functions.",
          link: "[ES] Learn More"
        },
        {
          title: "[ES] Trackie Tool Ecosystem",
          description: "[ES] Over 51 tools and micro-services, many developed in-house and integrated with the foundational AI, enabling specific functionalities and optimizing the experience via \"function calling\"."
        },
        {
          title: "[ES] Broad Platform Compatibility",
          description: "[ES] Primary focus on mobile (Android/iOS) and the hat, but the AI architecture (foundational and expandable) allows execution on Windows, Linux, and macOS, broadening R&D and niches."
        },
        {
          title: "[ES] Continuous Learning and Personalization",
          description: "[ES] Ability to learn from interactions, feed its database with new faces, and adapt to user preferences, becoming a truly personal assistant."
        }
      ]
    },
    aiEngineSection: {
        subtitle: "[ES] The Heart of Intelligence",
        title: "[ES] Trackie AI Engine: Flexible, Powerful, Adaptable",
        description: "[ES] Trackie's intelligence lies in a flexible architecture. A foundational AI, with its own algorithms and database trained for visual assistance, is the base that expands by integrating cutting-edge AI models, both online and offline. This approach, similar to Microsoft's Copilot concept, allows Trackie to be agnostic and use the best tool for each scenario, while actively developing its own optimized architecture, JuliusKaiser.",
        onlineTitle: "[ES] Cutting-Edge Online AI Models",
        offlineTitle: "[ES] Robust Offline Capability",
        offlineDescription: "[ES] Essential to ensure functionality anywhere. Models like Gemma, DeepSeek, and Llama run locally on the device, operating in conjunction with Trackie's foundational AI and database to maintain crucial assistance functions.",
        proprietaryTitle: "[ES] JuliusKaiser",
        proprietarySubtitle: "[ES] Coming Soon",
        developedAt: "[ES] Developed at SENAI \"Ettore Zanini\"",
        waitNews: "[ES] Stay Tuned for News",
        knowMoreJulius: "[ES] Learn more about JuliusKaiser soon",
        models: {
            gemini: {
                name: "[ES] Gemini 2.0 Flash",
                description: "[ES] Integrated for natural and contextual responses, leveraging Google's cutting-edge AI for voice interactions and multimodal processing, complementing Trackie's foundational AI.",
                features: ["[ES] Natural voice interaction", "[ES] Advanced contextual understanding", "[ES] Multimodal processing"]
            },
            openai: {
                name: "[ES] OpenAI Realtime API",
                description: "[ES] Focus on ultra-low latency interactions, ideal for immediate responses and fluid real-time dialogues. Uses OpenAI's latest models optimized for speed.",
                features: ["[ES] Real-time processing", "[ES] Low latency", "[ES] Dynamic interactions"]
            },
            namo: {
                name: "[ES] NAMO-SSLM",
                description: "[ES] Compact language model specialized in speech processing (Small Speech Language Model), optimized for efficiency in verbal interactions and voice commands.",
                features: ["[ES] Specialized in speech", "[ES] Compact and efficient", "[ES] Enhanced voice commands"]
            },
            gemma: {
                name: "[ES] Gemma (Google)",
                description: "[ES] Lightweight and efficient open-source model from Google, executed locally to ensure crucial visual assistance and interaction functionalities, in conjunction with Trackie's base AI, even without internet.",
                features: ["[ES] Optimized performance for edge", "[ES] Based on Gemini technology", "[ES] Offline availability"]
            },
            deepseek: {
                name: "[ES] DeepSeek Coder",
                description: "[ES] Open-source models with strong capabilities in code understanding and generation, potentially used for internal tools and offline logical reasoning, integrated into the Trackie system.",
                features: ["[ES] Strong in logic and code", "[ES] Active community", "[ES] Local execution"]
            },
            llama: {
                name: "[ES] Llama Family (Meta)",
                description: "[ES] Family of open-source models from Meta, offering various size and performance options for local processing, adapting to device capabilities and complementing Trackie's AI.",
                features: ["[ES] Wide range of sizes", "[ES] Scalable performance", "[ES] Robust offline use"]
            },
            julius: {
                name: "[ES] JuliusKaiser",
                description: "[ES] Our proprietary, super-intelligent AI architecture, actively developed by TrackWay at SENAI \"Ettore Zanini\". JuliusKaiser is the evolution of our foundational AI, built to work harmoniously with external models. It focuses on extreme environmental interpretation, deep understanding of complex scenarios, objects, and dynamic interactions, aiming for unparalleled efficiency, personalization, and synergy with our ecosystem.",
                features: ["[ES] Extreme Environmental Interpretation", "[ES] Advanced Scenario Comprehension", "[ES] TrackWay Hardware Optimization", "[ES] Superior Energy Efficiency", "[ES] Deep Personalization", "[ES] Continuous User-Focused Evolution"]
            }
        },
        logoAlt: {
          openai: "[ES] OpenAI Logo",
          deepseek: "[ES] DeepSeek Logo",
          meta: "[ES] Meta Llama Logo",
          googleGemini: "[ES] Google Gemini Logo",
          googleGemma: "[ES] Google Gemma Logo",
          namoSSLM: "[ES] NAMO-SSLM Logo",
          juliusKaiser: "[ES] JuliusKaiser AI Logo",
          spotway: "[ES] SpotWay Module Logo",
          raspway: "[ES] RaspWay Module Logo"
        },
        modelTypes: {
          online: "[ES] Online",
          offline: "[ES] Offline",
          proprietary: "[ES] Proprietary"
        }
    },
    visionSection: {
        subtitle: "[ES] [ OUR FUTURE VISION ]",
        title: "[ES] A Quantum Leap in Intelligent Assistance",
        description: "[ES] Trackie, with its intelligent ecosystem and hybrid AI architecture – uniting a proprietary AI base with trained algorithms and a specialized database, with the flexibility of cutting-edge AI models – represents an advancement in how AI can assist and empower individuals. Designed to be a discreet, powerful, and accessible companion, it offers a new level of independence and interaction with the world. We invite you to join TrackWay on this journey to make the future of intelligent assistance a reality for everyone.",
        joinButton: "[ES] Join the Trackie Revolution"
    },
    footer: {
        copyright: "[ES] TrackWay © {year}. All rights reserved.",
        projectInfo: "[ES] Conceptual project for demonstration.",
        projectLinkSenai: "[ES] Project Details (SENAI)"
    },
    general: {
        loading: "[ES] Loading...",
    }
  },
};
