import configparser
import os
from pathlib import Path


class ConfigManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_dir = Path(__file__).parent
        self.config_file = self.config_dir / 'config.ini'
        self.default_config = self.config_dir / 'default_config.ini'
        
        # Ensure required sections exist
        self.required_sections = ['PROMPTS', 'MODELS']
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create config file
        self._load_or_create_config()


    def _load_or_create_config(self):
        if not self.config_file.exists():
            # Copy default config if exists
            if self.default_config.exists():
                with open(self.default_config, 'r', encoding='utf-8') as f:
                    self.config.read_file(f)
            else:
                # Initialize with empty sections if no default config
                for section in self.required_sections:
                    self.config.add_section(section)
            # Save new config file
            self.save_config()
        else:
            self.config.read(self.config_file, encoding='utf-8')
            # Ensure all required sections exist
            for section in self.required_sections:
                if not self.config.has_section(section):
                    self.config.add_section(section)


    def save_config(self):
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)


    def get_prompts(self):
        if not self.config.has_section('PROMPTS'):
            self.config.add_section('PROMPTS')
            
        return {
            'system': self.config.get('PROMPTS', 'system_prompt', fallback=''),
            'system_default': self.config.get('PROMPTS', 'system_prompt_default', fallback=''),
            'user': self.config.get('PROMPTS', 'user_prompt', fallback=''),
            'user_default': self.config.get('PROMPTS', 'user_prompt_default', fallback='')
        }


    def set_prompt(self, prompt_type, text):
        if prompt_type not in ['system', 'user', 'system_default', 'user_default']:
            raise ValueError("Prompt type must be either 'system','user', 'system_default' or 'user_default'")
        
        if not self.config.has_section('PROMPTS'):
            self.config.add_section('PROMPTS')
            
        self.config.set('PROMPTS', f'{prompt_type}_prompt', text)
        self.save_config()

    ### Model List Retrieval and Manipulation Methods ###
    def get_models(self, provider=None):
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
        
        if provider:
            models = self.config.get('MODELS', f'{provider}_models', fallback='[]')
            return [v["name"] for v in eval(models)]
          # Convert string representation to list
        else:
            # Return all models combined
            openai_models = eval(self.config.get('MODELS', 'openai_models', fallback='[]'))
            groq_models = eval(self.config.get('MODELS', 'groq_models', fallback='[]'))
            return [v["name"] for v in openai_models + groq_models]


    def set_models(self, provider, models):
        if provider not in ['openai', 'groq']:
            raise ValueError("Provider must be either 'openai' or 'groq'")
        
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
            
        self.config.set('MODELS', f'{provider}_models', str(models))
        self.save_config()


    def add_model(self, provider, model_name, input_cost, output_cost):
        """
        Adds a new model to the provider's model list in the config.
        Example:
            self.add_model("openai", "gpt-6", 0.007, 0.014)
        """
        if provider not in ['openai', 'groq']:
            raise ValueError("Provider must be either 'openai' or 'groq'")
        
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')

        key = f'{provider}_models'

        # Safely load the existing models
        try:
            current_models = eval(self.config.get('MODELS', key, fallback='[]'))
        except Exception:
            current_models = []

        # Check if model already exists
        if any(m['name'] == model_name for m in current_models):
            print(f"Model '{model_name}' already exists for provider '{provider}'. Skipping.")
            return

        # Append the new model
        current_models.append({
            "name": model_name,
            "input": input_cost,
            "output": output_cost
        })

        # Save back to config
        self.set_models(provider, current_models)

    def remove_model(self, model_names):
        """
        Removes one or more models from the config for both providers (openai/groq).
        Example:
            self.remove_model("gpt-4o")
            self.remove_model(["gpt-5", "deepseek-r1-distill-llama-70b"])
        """
        if not isinstance(model_names, list):
            model_names = [model_names]

        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')

        for provider in ['openai', 'groq']:
            key = f'{provider}_models'

            # Load current models safely
            try:
                current_models = eval(self.config.get('MODELS', key, fallback='[]'))
            except Exception:
                current_models = []

            # Filter out models whose 'name' matches any in model_names
            updated_models = [m for m in current_models if m['name'] not in model_names]

            # Only update if something actually changed
            if len(updated_models) != len(current_models):
                self.set_models(provider, updated_models)
                print(f"✅ Removed models from '{provider}': {', '.join(set(model_names) - {m['name'] for m in updated_models})}")

        # Persist changes
        self.save_config()

                
    
    def reset_models(self):
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
            
        self.config.set('MODELS', 'openai_models', self.config.get('MODELS', 'openai_models_default', fallback='[]'))
        self.config.set('MODELS', 'groq_models', self.config.get('MODELS', 'groq_models_default', fallback='[]'))
        self.save_config()

    ### Current Model and API Management Methods ###    
    def get_current_model(self):
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
        return self.config.get('MODELS', 'current_model', fallback=None)
    

    def set_current_model(self, model_name):
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
        self.config.set('MODELS', 'current_model', model_name)
        self.save_config()


    def get_current_api(self):
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
        return self.config.get('MODELS', 'current_api', fallback="openai")
        

    def set_current_api(self, provider):
        if provider not in ['openai', 'groq']:
            raise ValueError("Provider must be either 'openai' or 'groq'")
        
        if not self.config.has_section('MODELS'):
            self.config.add_section('MODELS')
            
        self.config.set('MODELS', 'current_api', provider)
        self.save_config()
        
    ### Parameter Management Methods ###
    def get_parameters(self):
        if not self.config.has_section('PARAMETERS'):
            self.config.add_section('PARAMETERS')
        return {
            'teacher_name': self.config.get('PARAMETERS', 'teacher_name', fallback='LEHRER'),
            'teacher_name_options': self.config.get('PARAMETERS', 'teacher_name', fallback='LEHRER'),
            'group_id': self.config.get('PARAMETERS', 'group_id', fallback='B1'),
            'num_pupils': self.config.getint('PARAMETERS', 'num_pupils', fallback=25),
            'teacher_name_default': self.config.get('PARAMETERS', 'teacher_name_default', fallback='LEHRER'),
            'group_id_default': self.config.get('PARAMETERS', 'group_id_default', fallback='B1'),
            'num_pupils_default': self.config.getint('PARAMETERS', 'num_pupils_default', fallback=25)
        }
    
    
    def set_parameter(self, key, value):
        if key not in ['teacher_name', 'teacher_name_options', 'group_id', 'num_pupils']:
            raise ValueError("Parameter key must be either 'teacher_name', 'teacher_name_options', 'group_id' or 'num_pupils'")
        
        if not self.config.has_section('PARAMETERS'):
            self.config.add_section('PARAMETERS')
            
        self.config.set('PARAMETERS', key, str(value))
        self.save_config()

    ### Localization Management Methods ###
    def get_localization(self):
        if not self.config.has_section('LOCALIZATION'):
            self.config.add_section('LOCALIZATION')
        return {
            'default_language': self.config.get('LOCALIZATION', 'default_language', fallback='en'),
            'current_language': self.config.get('LOCALIZATION', 'current_language', fallback='en'),
            'default_language_default': self.config.get('LOCALIZATION', 'default_language_default', fallback='en'),
            'current_language_default': self.config.get('LOCALIZATION', 'current_language_default', fallback='en'),
        }
    

    def set_localization(self, key, value):
        if value not in ['de', 'en']:
            raise ValueError("Localization must be either 'de' or 'en'")
        if key not in ['default_language', 'current_language']:
            raise ValueError("Localization must be either 'de' or 'en'")
        
        if not self.config.has_section('LOCALIZATION'):
            self.config.add_section('LOCALIZATION')
            
        self.config.set('LOCALIZATION', key, str(value))
        self.save_config()

    ### Pricing Prediction Helper Method ###
    def get_api_pricing(self):
        """Returns pricing for different APIs and models"""
        pricing = {}
        for provider in ['openai', 'groq']:
            key = f"{provider}_models"
            models_str = self.config.get('MODELS', key, fallback='[]')
            try:
                models = eval(models_str)  # convert to list of dicts
            except Exception:
                models = []

            # Build dictionary of model: {input, output}
            provider_pricing = {
                m['name']: {'input': m['input'], 'output': m['output']}
                for m in models if all(k in m for k in ('name', 'input', 'output'))
            }

            pricing[provider] = provider_pricing

        return pricing
