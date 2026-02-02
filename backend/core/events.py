"""
Core events handlers and EventEmitter
"""

from typing import Dict, List, Callable, Any
import asyncio


class EventEmitter:
    """
    Event emitter per comunicazione tra componenti
    Supporta callback sync e async
    """
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._once_listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable):
        """Registra listener per evento"""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
        return self
    
    def once(self, event: str, callback: Callable):
        """Registra listener che si rimuove dopo prima esecuzione"""
        if event not in self._once_listeners:
            self._once_listeners[event] = []
        self._once_listeners[event].append(callback)
        return self
    
    def off(self, event: str, callback: Callable = None):
        """Rimuove listener"""
        if callback is None:
            self._listeners.pop(event, None)
            self._once_listeners.pop(event, None)
        else:
            if event in self._listeners:
                self._listeners[event] = [cb for cb in self._listeners[event] if cb != callback]
            if event in self._once_listeners:
                self._once_listeners[event] = [cb for cb in self._once_listeners[event] if cb != callback]
        return self
    
    async def emit(self, event: str, *args, **kwargs) -> List[Any]:
        """Emette evento e chiama tutti i listener"""
        results = []
        
        # Regular listeners
        for callback in self._listeners.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # Once listeners (rimuovi dopo esecuzione)
        once_callbacks = self._once_listeners.pop(event, [])
        for callback in once_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def emit_sync(self, event: str, *args, **kwargs) -> List[Any]:
        """Emette evento in modo sincrono (solo callback sync)"""
        results = []
        
        for callback in self._listeners.get(event, []):
            if not asyncio.iscoroutinefunction(callback):
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        return results
    
    def listeners(self, event: str) -> List[Callable]:
        """Lista listener per evento"""
        return self._listeners.get(event, []) + self._once_listeners.get(event, [])
    
    def event_names(self) -> List[str]:
        """Lista nomi eventi registrati"""
        return list(set(list(self._listeners.keys()) + list(self._once_listeners.keys())))


# Global event emitter
global_events = EventEmitter()


async def startup_event():
    """Run on application startup"""
    await global_events.emit("startup")


async def shutdown_event():
    """Run on application shutdown"""
    await global_events.emit("shutdown")
