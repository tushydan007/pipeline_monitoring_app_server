from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class Command(BaseCommand):
    help = 'Make email field unique for all existing users and handle duplicates'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No changes will be made'))
        
        # Find duplicate emails
        from django.db.models import Count
        duplicates = User.objects.values('email').annotate(
            count=Count('email')
        ).filter(count__gt=1, email__isnull=False).exclude(email='')
        
        if duplicates.exists():
            self.stdout.write(self.style.ERROR(
                f'Found {duplicates.count()} duplicate email addresses:'
            ))
            for dup in duplicates:
                users = User.objects.filter(email=dup['email'])
                self.stdout.write(f"  Email: {dup['email']} - {users.count()} users")
                for user in users:
                    self.stdout.write(f"    - {user.username} (ID: {user.id})")
            
            if not dry_run:
                self.stdout.write(self.style.WARNING(
                    'Please resolve duplicate emails manually before running migration'
                ))
            return
        
        # Update null/empty emails
        users_without_email = User.objects.filter(
            models.Q(email__isnull=True) | models.Q(email='')
        )
        
        if users_without_email.exists():
            self.stdout.write(self.style.WARNING(
                f'Found {users_without_email.count()} users without email addresses'
            ))
            if not dry_run:
                for user in users_without_email:
                    # Generate a unique email based on username
                    base_email = f"{user.username}@example.com"
                    counter = 1
                    while User.objects.filter(email=base_email).exists():
                        base_email = f"{user.username}{counter}@example.com"
                        counter += 1
                    user.email = base_email
                    user.save()
                    self.stdout.write(f"  Set email for {user.username}: {base_email}")
        
        # Normalize emails to lowercase
        users_with_email = User.objects.exclude(email__isnull=True).exclude(email='')
        if not dry_run:
            updated = 0
            for user in users_with_email:
                old_email = user.email
                new_email = old_email.lower().strip()
                if old_email != new_email:
                    user.email = new_email
                    user.save()
                    updated += 1
            if updated > 0:
                self.stdout.write(self.style.SUCCESS(
                    f'Normalized {updated} email addresses to lowercase'
                ))
        
        if dry_run:
            self.stdout.write(self.style.SUCCESS('Dry run completed'))
        else:
            self.stdout.write(self.style.SUCCESS(
                'Email addresses are ready for unique constraint'
            ))

