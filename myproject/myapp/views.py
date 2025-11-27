from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from .forms import TodoForm
from .models import Todo


def home(request):
    todos = Todo.objects.order_by("-created_at")
    form = TodoForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            form.save()
            return redirect("home")
    else:
        form = TodoForm()

    return render(
        request,
        "home.html",
        {
            "form": form,
            "todos": todos,
        },
    )


@require_http_methods(["POST"])
def toggle_todo(request, pk):
    todo = get_object_or_404(Todo, pk=pk)
    todo.is_completed = not todo.is_completed
    todo.save(update_fields=["is_completed", "updated_at"])
    return redirect("home")


@require_http_methods(["POST"])
def delete_todo(request, pk):
    todo = get_object_or_404(Todo, pk=pk)
    todo.delete()
    return redirect("home")
