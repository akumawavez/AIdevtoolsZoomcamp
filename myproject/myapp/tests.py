from django.test import TestCase
from django.urls import reverse

from .models import Todo


class TodoViewsTests(TestCase):
    def test_home_displays_todos(self):
        Todo.objects.create(title="Sample task")

        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sample task")

    def test_home_creates_todo(self):
        response = self.client.post(reverse("home"), {"title": "New task", "description": "Details"})

        self.assertEqual(response.status_code, 302)
        self.assertTrue(Todo.objects.filter(title="New task").exists())

    def test_toggle_todo_changes_state(self):
        todo = Todo.objects.create(title="Toggle me", is_completed=False)

        self.client.post(reverse("toggle_todo", args=[todo.pk]))
        todo.refresh_from_db()

        self.assertTrue(todo.is_completed)

    def test_delete_todo_removes_instance(self):
        todo = Todo.objects.create(title="Delete me")

        self.client.post(reverse("delete_todo", args=[todo.pk]))

        self.assertFalse(Todo.objects.filter(pk=todo.pk).exists())
