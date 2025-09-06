#!/usr/bin/env python3
"""
Automated Tuning Application Script for Project PokerMind.

This script reads tuning_suggestions.json and programmatically updates
the config/agent_config.yaml file while preserving comments and structure.

Implements the autonomous tuning system required by the Ultimate Intelligence Protocol.
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ruamel.yaml for comment-preserving YAML editing
try:
    from ruamel.yaml import YAML
    YAML_AVAILABLE = True
except ImportError:
    import yaml
    YAML_AVAILABLE = False
    print("Warning: ruamel.yaml not available. Comments will not be preserved.")


class AutomatedTuner:
    """
    Automated configuration tuning system.
    
    Reads machine-readable tuning suggestions and applies them to the 
    agent configuration while preserving structure and comments.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize the automated tuner.
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize YAML handler
        if YAML_AVAILABLE:
            self.yaml = YAML()
            self.yaml.preserve_quotes = True
            self.yaml.width = 1000
            self.yaml.indent(mapping=2, sequence=2, offset=0)
        else:
            self.yaml = None
        
        self.backup_path = None
        
    def load_tuning_suggestions(self, suggestions_path: str = "tuning_suggestions.json") -> Dict[str, Any]:
        """
        Load tuning suggestions from JSON file.
        
        Args:
            suggestions_path: Path to tuning suggestions file
            
        Returns:
            Dictionary containing tuning suggestions
        """
        try:
            with open(suggestions_path, 'r') as f:
                suggestions = json.load(f)
            
            self.logger.info(f"Loaded tuning suggestions from {suggestions_path}")
            return suggestions
            
        except FileNotFoundError:
            self.logger.error(f"Tuning suggestions file not found: {suggestions_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in tuning suggestions: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading tuning suggestions: {e}")
            return {}
    
    def backup_current_config(self) -> str:
        """
        Create a backup of the current configuration.
        
        Returns:
            Path to the backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"agent_config_backup_{timestamp}.yaml"
        backup_dir = os.path.join(os.path.dirname(self.config_path), "backups")
        
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        self.backup_path = os.path.join(backup_dir, backup_filename)
        
        try:
            with open(self.config_path, 'r') as source:
                with open(self.backup_path, 'w') as backup:
                    backup.write(source.read())
            
            self.logger.info(f"Configuration backed up to: {self.backup_path}")
            return self.backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def load_current_config(self) -> Dict[str, Any]:
        """
        Load current agent configuration.
        
        Returns:
            Current configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                if YAML_AVAILABLE:
                    config = self.yaml.load(f)
                else:
                    config = yaml.safe_load(f)
            
            return config or {}
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
    
    def apply_parameter_changes(self, config: Dict[str, Any], suggested_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter changes to configuration.
        
        Args:
            config: Current configuration
            suggested_changes: Dictionary of parameter changes
            
        Returns:
            Updated configuration
        """
        changes_applied = {}
        
        for param_path, new_value in suggested_changes.items():
            try:
                # Split parameter path (e.g., "synthesizer.gto_weight")
                path_parts = param_path.split('.')
                
                # Navigate to the correct section
                current_section = config
                for part in path_parts[:-1]:
                    if part not in current_section:
                        current_section[part] = {}
                    current_section = current_section[part]
                
                # Get current value for logging
                old_value = current_section.get(path_parts[-1], "not_set")
                
                # Apply the change
                current_section[path_parts[-1]] = new_value
                
                changes_applied[param_path] = {
                    "old_value": old_value,
                    "new_value": new_value
                }
                
                self.logger.info(f"Applied change: {param_path} = {old_value} -> {new_value}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply change for {param_path}: {e}")
        
        return changes_applied
    
    def save_updated_config(self, config: Dict[str, Any]) -> bool:
        """
        Save updated configuration to file.
        
        Args:
            config: Updated configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                if YAML_AVAILABLE:
                    self.yaml.dump(config, f)
                else:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Updated configuration saved to: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration after changes.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required sections exist
            required_sections = ['synthesizer', 'gto_core', 'player_style', 'models']
            
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate synthesizer weights sum to approximately 1.0
            module_weights = config.get('synthesizer', {}).get('module_weights', {})
            if module_weights:
                weight_sum = sum(module_weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    self.logger.warning(f"Module weights sum to {weight_sum:.3f}, should be 1.0")
                    # Auto-normalize weights
                    for key in module_weights:
                        module_weights[key] = module_weights[key] / weight_sum
                    self.logger.info("Auto-normalized module weights")
            
            # Validate confidence thresholds are in valid range
            confidence_threshold = config.get('gto_core', {}).get('confidence_threshold', 0.7)
            if not 0.0 <= confidence_threshold <= 1.0:
                self.logger.error(f"Invalid confidence threshold: {confidence_threshold}")
                return False
            
            # Validate player style parameters
            player_style = config.get('player_style', {})
            for param in ['aggression', 'tightness']:
                value = player_style.get(param, 0.5)
                if not 0.0 <= value <= 1.0:
                    self.logger.error(f"Invalid player style parameter {param}: {value}")
                    return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def generate_change_report(self, suggestions: Dict[str, Any], changes_applied: Dict[str, Any]) -> str:
        """
        Generate a report of changes applied.
        
        Args:
            suggestions: Original tuning suggestions
            changes_applied: Dictionary of changes that were applied
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("AUTOMATED TUNING CHANGES REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Applied at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Backup saved to: {self.backup_path}")
        report_lines.append("")
        
        # Analysis context
        sample_size = suggestions.get("sample_size", 0)
        confidence = suggestions.get("analysis_confidence", 0.0)
        report_lines.append(f"Analysis based on: {sample_size} hands")
        report_lines.append(f"Analysis confidence: {confidence:.1%}")
        report_lines.append("")
        
        # Changes applied
        report_lines.append("CHANGES APPLIED")
        report_lines.append("-" * 20)
        
        if not changes_applied:
            report_lines.append("No changes were applied.")
        else:
            priority_order = suggestions.get("priority_order", list(changes_applied.keys()))
            
            for i, param in enumerate(priority_order, 1):
                if param in changes_applied:
                    change = changes_applied[param]
                    report_lines.append(f"{i}. {param}")
                    report_lines.append(f"   {change['old_value']} → {change['new_value']}")
                    
                    # Add expected impact if available
                    expected_impact = suggestions.get("expected_impact", {}).get(param, {})
                    if expected_impact:
                        improvement = expected_impact.get("performance_improvement", "unknown")
                        report_lines.append(f"   Expected improvement: {improvement}")
                    
                    report_lines.append("")
        
        # Strategic insights
        strategic_adjustments = suggestions.get("strategic_adjustments", {})
        if strategic_adjustments:
            report_lines.append("STRATEGIC INSIGHTS")
            report_lines.append("-" * 18)
            
            for category, analysis in strategic_adjustments.items():
                if isinstance(analysis, dict):
                    report_lines.append(f"• {category.replace('_', ' ').title()}:")
                    for key, value in analysis.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {key.replace('_', ' ')}: {value:.3f}")
                        else:
                            report_lines.append(f"  {key.replace('_', ' ')}: {value}")
                    report_lines.append("")
        
        # Implementation notes
        impl_notes = suggestions.get("implementation_notes", {})
        if impl_notes:
            report_lines.append("IMPLEMENTATION NOTES")
            report_lines.append("-" * 20)
            for key, note in impl_notes.items():
                report_lines.append(f"• {note}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def apply_tuning_suggestions(self, suggestions_path: str = "tuning_suggestions.json", 
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Main method to apply tuning suggestions.
        
        Args:
            suggestions_path: Path to tuning suggestions file
            dry_run: If True, don't actually modify files
            
        Returns:
            Dictionary with results of the tuning process
        """
        result = {
            "success": False,
            "changes_applied": {},
            "backup_path": None,
            "error": None
        }
        
        try:
            # Load tuning suggestions
            suggestions = self.load_tuning_suggestions(suggestions_path)
            if not suggestions:
                result["error"] = "No valid tuning suggestions found"
                return result
            
            # Load current configuration
            config = self.load_current_config()
            if not config:
                result["error"] = "Failed to load current configuration"
                return result
            
            if not dry_run:
                # Create backup
                result["backup_path"] = self.backup_current_config()
            
            # Apply parameter changes
            suggested_changes = suggestions.get("suggested_parameter_changes", {})
            if not suggested_changes:
                self.logger.info("No parameter changes suggested")
                result["success"] = True
                return result
            
            changes_applied = self.apply_parameter_changes(config, suggested_changes)
            result["changes_applied"] = changes_applied
            
            # Validate updated configuration
            if not self.validate_configuration(config):
                result["error"] = "Configuration validation failed"
                return result
            
            if not dry_run:
                # Save updated configuration
                if not self.save_updated_config(config):
                    result["error"] = "Failed to save updated configuration"
                    return result
            
            # Generate and save change report
            report = self.generate_change_report(suggestions, changes_applied)
            
            if not dry_run:
                report_path = f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                report_dir = os.path.join(os.path.dirname(self.config_path), "reports")
                os.makedirs(report_dir, exist_ok=True)
                
                full_report_path = os.path.join(report_dir, report_path)
                with open(full_report_path, 'w') as f:
                    f.write(report)
                
                self.logger.info(f"Tuning report saved to: {full_report_path}")
            
            # Print report summary
            print(report)
            
            result["success"] = True
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying tuning suggestions: {e}")
            result["error"] = str(e)
            return result
    
    def rollback_changes(self, backup_path: str = None) -> bool:
        """
        Rollback changes by restoring from backup.
        
        Args:
            backup_path: Path to backup file (uses last backup if not specified)
            
        Returns:
            True if successful, False otherwise
        """
        if not backup_path:
            backup_path = self.backup_path
        
        if not backup_path or not os.path.exists(backup_path):
            self.logger.error("No backup file available for rollback")
            return False
        
        try:
            with open(backup_path, 'r') as backup:
                with open(self.config_path, 'w') as config:
                    config.write(backup.read())
            
            self.logger.info(f"Configuration rolled back from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback configuration: {e}")
            return False


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Apply automated tuning suggestions to PokerMind agent")
    parser.add_argument(
        "--suggestions",
        type=str,
        default="tuning_suggestions.json",
        help="Path to tuning suggestions JSON file"
    )
    parser.add_argument(
        "--config",
        type=str, 
        default="config/agent_config.yaml",
        help="Path to agent configuration YAML file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually applying them"
    )
    parser.add_argument(
        "--rollback",
        type=str,
        help="Rollback changes using specified backup file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Project PokerMind - Automated Tuning System")
    print("=" * 60)
    
    # Initialize tuner
    tuner = AutomatedTuner(args.config)
    
    try:
        if args.rollback:
            # Rollback mode
            print(f"Rolling back configuration from: {args.rollback}")
            success = tuner.rollback_changes(args.rollback)
            if success:
                print("✓ Configuration successfully rolled back")
            else:
                print("✗ Rollback failed")
                sys.exit(1)
        else:
            # Apply tuning mode
            if args.dry_run:
                print("DRY RUN MODE - No changes will be applied")
                print("")
            
            result = tuner.apply_tuning_suggestions(args.suggestions, args.dry_run)
            
            if result["success"]:
                changes_count = len(result["changes_applied"])
                print(f"\n✓ Tuning application completed successfully!")
                print(f"  Applied {changes_count} parameter changes")
                if result["backup_path"]:
                    print(f"  Backup saved to: {result['backup_path']}")
            else:
                print(f"\n✗ Tuning application failed: {result['error']}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nTuning application interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()